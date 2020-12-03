from flask import Flask, url_for

app = Flask(__name__)

@app.route('/')
def hello():
    return "<img src='{{ url_for('static', filename='title.png') }}'>"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port="8080")