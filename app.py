from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from chat import get_response
import requests
import os
from dotenv import load_dotenv
load_dotenv()

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

@app.post("/wa-webhook")
def post_wa_webhook():
    params = request.get_json()
    if params['entry'] and params['entry'][0]['changes'] and params['entry'][0]['changes'][0] and params['entry'][0]['changes'][0]['value']['messages'] and params['entry'][0]['changes'][0]['value']['messages'][0]:
        phone_number_id = params['entry'][0]['changes'][0]['value']['metadata']['phone_number_id']
        phone_number_from = params['entry'][0]['changes'][0]['value']['messages'][0]['from']
        msg_body = params['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
        token = os.getenv('WA_API_KEY')
        url = 'https://graph.facebook.com/v12.0/' + phone_number_id + '/messages?access_token=' + token
        dict_to_send = {
            'messaging_product': "whatsapp",
            'to': phone_number_from,
            'text': { 'body': "Ack: " + msg_body },
        }
        headers = {'Content-Type': 'application/json'}

        try:
            requests.post(url, json=dict_to_send, headers=headers)
        except KeyError:
            print('error')

        return make_response(jsonify({'message': 'berhasil'}), 200)
    else:
        return make_response(jsonify({'message': 'gagal'}), 404)

@app.get("/wa-webhook")
def get_wa_webhook():
    verify_token = os.getenv('WA_TOKEN_VERIF')

    params = request.args
    mode = params.get('hub.mode')
    token = params.get('hub.verify_token')
    challenge = params.get('hub.challenge')

    if mode != None and token != None:
        if mode == "subscribe" and token == verify_token:
            return make_response(challenge, 200)

    return make_response(jsonify({'message': 'gagal'}), 403)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="3000")
