from flask import Flask, url_for, render_template, request
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import sys
import time
import json
from urllib import parse
import re
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

api_key = 'RGAPI-6e66ed4a-2403-42d0-8571-96c89c0a2e95'
params = {'api_key': api_key, 'queue' : 420}
host = 'https://kr.api.riotgames.com'

def getParticipantId(partIdn, name):
    pid = 0
    p = partIdn
    p = str(p)
    pss = p.replace("'", "\"")
    p_dict = json.loads(pss)


    for i in range(len(p_dict)):
        if p_dict[i]['player']['summonerName'].replace(' ','') == name.replace(' ',''):
            #print (name)
            #print (p_dict[i]['player']['summonerName'])
            pid = p_dict[i]['participantId']
            break
    #print (pid)
    return pid

def isHangul(text):
    #Check the Python Version
    pyVer3 =  sys.version_info >= (3, 0)

    if pyVer3 : # for Ver 3 or later
        encText = text
    else: # for Ver 2.x
        if type(text) is not unicode:
            encText = text.decode('utf-8')
        else:
            encText = text

    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0


def get_accountid(summonerName):
    if isHangul(summonerName) is True:
        summonerName = parse.quote(summonerName)

    endpoint = '/lol/summoner/v4/summoners/by-name/'+summonerName
    #print (endpoint)
    #endpoint = endpoint.format(summonerName=summonerName)
    res = requests.get(host + endpoint, params= params)
    while(True):
        try:
            res.json()['status']
            if res.json()['status']['status_code'] == 404:
                return 0
            print('sleep 10 sec')
            time.sleep(10)
        except:
            a = 1
            return res.json()['accountId']


def get_match_list(account_id, begin_time):
    endpoint = '/lol/match/v4/matchlists/by-account/{encryptedAccountId}'
    endpoint = endpoint.format(encryptedAccountId=account_id)


    r = requests.get(host + endpoint, params=params)
    #print(r.json())
    if r.status_code != 200:
        print(r.json())

    if r.status_code == 200: # response가 정상이면 바로 맨 밑으로 이동하여 정상적으로 코드 실행
        pass
    if r.status_code ==404:
        return "404"

    elif r.status_code == 429:
        print('api cost full : infinite loop start')
        start_time = time.time()

        while True: # 429error가 끝날 때까지 무한 루프
            if r.status_code == 429 or r.status_code == 401  or r.status_code == 504:

                print('try 10 second wait time')
                time.sleep(10)

                r = requests.get(api_url,  params = params)

                #print(r.json())

            elif r.status_code == 200: #다시 response 200이면 loop escape
                print('total wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
    elif r.status_code == 503 or r.status_code == 504: # 잠시 서비스를 이용하지 못하는 에러
        print('service available error')
        start_time = time.time()

        while True:
            if r.status_code == 503 or r.status_code == 429 or r.status_code == 401 or r.status_code == 504:
                print('try 10 second wait time')

                time.sleep(10)
                r = requests.get(api_url,  params = params)
                #print(r.json())

            elif r.status_code == 200: # 똑같이 response가 정상이면 loop escape
                print('total error wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
    elif r.status_code == 403: # api갱신이 필요
        print('you need api renewal')
        print('break')

    match_ids = []
    for id in r.json()['matches']:
        if(id['timestamp'] > begin_time):
            match_ids.append(id['gameId'])
    return match_ids

def getGameData(gameId):
    api_url='https://kr.api.riotgames.com/lol/match/v4/matches/' + str(gameId)
    r = requests.get(api_url, params = params)
    #print (r.json()['gameDuration'])
    if r.status_code != 200:
        print (r.json())

    if r.status_code == 200: # response가 정상이면 바로 맨 밑으로 이동하여 정상적으로 코드 실행
        pass
    if r.status_code ==404:
        return None

    elif r.status_code == 429:
        print('api cost full : infinite loop start')
        start_time = time.time()

        while True: # 429error가 끝날 때까지 무한 루프
            if r.status_code == 429 or r.status_code == 401  or r.status_code == 504:

                print('try 10 second wait time')
                time.sleep(10)

                r = requests.get(api_url,  params = params)

                #print(r.json())

            elif r.status_code == 200: #다시 response 200이면 loop escape
                print('total wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
    elif r.status_code == 503 or r.status_code == 504: # 잠시 서비스를 이용하지 못하는 에러
        print('service available error')
        start_time = time.time()

        while True:
            if r.status_code == 503 or r.status_code == 429 or r.status_code == 401 or r.status_code == 504:
                print('try 10 second wait time')

                time.sleep(10)
                r = requests.get(api_url,  params = params)
                #print(r.json())

            elif r.status_code == 200: # 똑같이 response가 정상이면 loop escape
                print('total error wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
    elif r.status_code == 403: # api갱신이 필요
        print('you need api renewal')
        print('break')

    #print (mat)
    return r.json()

def get_user_matchdata(name, due):
    now = int(time.time()*1000.0)
    begin_time = now - due * 24 * 60 * 60 * 1000
    print ("sdf", name)

    account_id = get_accountid(name)
    if account_id == 0:
        return None
    match_list = get_match_list(account_id, begin_time)

    result = []

    for id in match_list:
        match_data = getGameData(id)
        p_id = getParticipantId(match_data['participantIdentities'], name)
        match_stat = match_data['participants'][p_id-1]['stats']
        game_duration = match_data['gameDuration']
        if(game_duration < 5 * 60):
            continue
        champ_level = match_stat['champLevel']
        gold_earned = match_stat['goldEarned']
        gold_spent = match_stat['goldSpent']
        deaths = match_stat['deaths']
        kills = match_stat['kills']
        assists = match_stat['assists']
        #print (str(kills),str(deaths),str(assists))
        game_participation = champ_level / (game_duration / 60)
        gold_spent_ratio = abs((gold_earned / (1+gold_spent)) - 1)
        death_ratio = deaths / champ_level
        kda = 0
        if(kills == 0 and deaths == 0):
            kda = 0
        elif(kills != 0 and deaths == 0):
            kda = (kills + assists)
        else:
            kda = ((kills + assists) / deaths )

        dup_item = 0
        items = []
        for i in range(6):
            items.append(match_stat['item'+str(i)])
        for i in range(6):
            if(items[i] == 0):
                continue
            else:
                dup = (items.count(items[i])-1) ** 2
                if(dup > dup_item):
                    dup_item = dup

        result.append({
          'dup_item': dup_item,
          'kda': kda,
          'game_participation': game_participation,
          'gold_spent_ratio': gold_spent_ratio,
          'death_ratio': death_ratio,
        })
    return result

df  = pd.read_csv('./최종사용데이터/train_data.csv', encoding='CP949')
del df['Unnamed: 0']
y = df[['class']]
x = df[['dup_item','kda', 'game_participation', 'gold_spent_ratio','death_ratio']]
#x = df[['dup_item', 'kda', 'gold_spent_ratio']]

def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()


app = Flask(__name__)

@app.route('/')
def hello(_name = None, _message = None):
    return render_template('html.html', _name = _name, _message = _message)

@app.route('/search')
def do_(_name = None, _message = None):
    global x
    global y
    name = request.args.get('name').encode('utf-8').decode()
    print (name)
    user = get_user_matchdata(name, 7)
    if user is None:
        return render_template('html.html', _message="존재하지 않는 소환사명입니다.", _name = None)
    p_d = pd.DataFrame(get_user_matchdata(name, 7))##date 설정 최근 n일
    if p_d.empty is True:
        return render_template('html.html', _message="최근 일주일간 플레이한 솔로 랭킹 게임이 없습니다.", _name = None)
    x = pd.concat([x, p_d])
    x_dup = pd.DataFrame(x['dup_item'])
    x_dup[:] = StandardScaler().fit_transform(x_dup[:])
    x['dup_item'] = x_dup[:]
    p_d = x[-len(p_d):]
    p_d['dup_item'] = x_dup[-len(p_d):]
    x = x[:-len(p_d)]

    svm = SVC(kernel='poly', class_weight = 'balanced')
    svm.fit(x, np.ravel(y,order='C'))
    pred = svm.predict(p_d)
    print ("최근 일주일 솔로 랭킹 게임 수 : ", len(pred), "게임")
    print ("최근 탐지된 비매너 게임 : ", np.count_nonzero(pred == 1), "게임")
    print (name,"님은 ",str(100* np.count_nonzero(pred == 1)/len(pred)),"% 트롤")


    return render_template('html.html', _name = name, game_n = len(pred), troll_n = np.count_nonzero(pred == 1), percent = format(100* np.count_nonzero(pred == 1)/len(pred),".2f"))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port="8080")