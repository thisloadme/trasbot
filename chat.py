import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from typo_detector import tokenize_correct_typo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

model_state = data['model_state']
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Trasbot'

def get_not_understanding_message():
    message = random.choice([
        "Maaf, aku tidak paham maksudmu",
        "Maaf, aku tidak bisa menangkap maksudmu",
        "Maaf, aku belum mengerti",
        "Maaf, bisa beri aku penjelasan lebih detail?"
    ])
    return {'tags': None, 'message':message, 'random':None}

def get_response(message):
    sentence = tokenize_correct_typo(message, all_words)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tags']:
                item_random = None
                if len(intent['randoms']) > 0:
                    item_random = random.choices(intent['randoms'], k=5)
                    item_random = ', '.join(item_random)

                return {'tags': tag, 'message':random.choice(intent['responses']), 'random':item_random}
    else:
        return get_not_understanding_message()

if __name__ == '__main__':
    print("Halo dengan Trasbot disini, kamu bisa menanyakan kepadaku apapun tentang Traspac.")
    while True:
        message = input('You: ')
        if message == 'quit':
            break

        resp = get_response(message)
        resp_tags = resp['tags']
        resp_message = resp['message']
        resp_message = resp_message.replace("(sameres)", message.capitalize())
        resp_random = resp['random']
        
        print(bot_name + ': ' + resp_message + ' ' + (resp_random if resp_random != None else ''))
        
        if resp_tags == 'terimakasih':
            break