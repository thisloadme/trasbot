import os
import json
from nlp_utils import tokenize, bag_of_words, remove_stopwords_indo, slang_word_meaning
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_lstm import LSTM

current_dir = os.getcwd()
ignore_words = ['?', '!', '<', '>', '.', ',']

with open(current_dir + '/intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tags']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        w = [slang_word_meaning(w_) for w_ in w if w_ not in ignore_words]
        all_words.extend(w)

        xy.append((w, tag))

all_words = [w.lower() for w in all_words if w not in ignore_words]
# all_words = remove_stopwords_indo(all_words)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words, True)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
batch_size = 16
hidden_size = 128
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 100
min_error = 0.00001

dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

loss_values = []
for epoch in range(num_epochs):
    num_correct = 0
    num_samples = 0

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        # print(words)
        # print(labels)
        # print(len(all_words))
        # print(words.shape)

        outputs = model(words)
        _, predictions = outputs.max(1)
        num_correct += (predictions == labels).sum()
        num_samples += predictions.size(0)
        # print(epoch)
        # print(outputs)
        # print(outputs.shape)
        # print(labels)
        # print(labels.shape)
        # print(predictions)
        # print(predictions.shape)
        # print(num_correct)
        # print(num_correct.shape)
        # exit()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

    # print((epoch+1))
    # print(num_correct)
    # print(num_samples)
    # print(float(num_correct)/float(num_samples))
    # exit()
    # if loss.item() <= min_error:
    #     break

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}, accuracy={float(num_correct)/float(num_samples)*100:.2f}')

# exit()
print(f'final loss={loss.item():.4f}')

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'{num_correct} / {num_samples} correct, with accuracy {float(num_correct)/float(num_samples)*100:.2}')
    model.train()

# check_accuracy(train_loader, model)

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = current_dir + '/data_lstm.pth'
torch.save(data, FILE)

plt.plot(loss_values)
plt.show()

print(f'training complete')