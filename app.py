from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)

@app.get("/")
def home():
    return 'Hello world'

@app.get("/predict")
def predict():
    text = request.args.get('message')

    response = get_response(text)
    message = {"answer": response['message'] + '' + ((' ' + response['random']) if response['random'] != None else '')}
    return jsonify(message)

@app.post("/chat")
def chat():
    text = request.get_json().get('message')

    response = get_response(text)
    message = {"answer": response['message'] + '' + ((' ' + response['random']) if response['random'] != None else '')}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
