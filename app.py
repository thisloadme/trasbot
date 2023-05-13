from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# from chat import get_response

app = Flask(__name__)
CORS(app)

# @app.get("/predict")
# def predict():
#     text = request.get_json().get('message')

#     response = get_response(text)
#     message = {"answer": response['message']}
#     return jsonify(message)

@app.get('/')
def form():
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)