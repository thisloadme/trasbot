import os
import random
import json
import torch
from model import NeuralNet
from nlp_utils import bag_of_words, remove_stopwords_indo
from utility import tokenize_correct_typo_slang

current_dir = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(current_dir + '/intents.json', 'r') as f:
    intents = json.load(f)

FILE = current_dir + '/data.pth'
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
    tokenized_sentence = tokenize_correct_typo_slang(message, all_words)
    tokenized_sentence = remove_stopwords_indo(tokenized_sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # print(tokenized_sentence)
    # print(tag)
    # print(prob)

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag != intent['tags']:
                continue
            
            options = None
            if len(intent['randoms']) > 0:
                options = random.choices(intent['randoms'], k=5)
                options = ', '.join(options)

            response = random.choice(intent['responses'])
            response = response.replace("(sameres)", message.capitalize())

            return {'tags': tag, 'message':response, 'random':options}
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
        resp_random = resp['random']
        
        print(bot_name + ': ' + resp_message + ' ' + (resp_random if resp_random != None else ''))
        
        if resp_tags == 'terimakasih':
            break