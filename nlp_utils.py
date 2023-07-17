import os
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import numpy as np
import json

current_dir = os.getcwd()

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def sentence_tokenize(paragraph):
    return nltk.sent_tokenize(paragraph)

def stem(word):
    return stemmer.stem(word.lower())

def remove_stopwords_indo(tokenized_sentence):
    with open(current_dir + '/kamus/combined_stop_words.txt', 'r') as f:
        lines = f.readlines()

    stopwords_indo = [w.replace("\n", '') for w in lines]
    return [w for w in tokenized_sentence if w not in stopwords_indo]

def slang_word_meaning(word):
    with open(current_dir + '/kamus/combined_slang_words.txt', 'r') as f:
        dict = json.load(f)
    
    return dict[word] if word in dict else word

def bag_of_words(tokenized_sentence, all_words, is_without_stem=False):
    if is_without_stem:
        tokenized_sentence = [w.lower() for w in tokenized_sentence]
    else:
        tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

def word_vec(tokenized_sentence, all_words):
    bag = np.zeros(100, dtype=np.int32)
    for idx, w in enumerate(tokenized_sentence):
        if w in all_words:
            bag[idx] = all_words.index(w)
    
    return bag

def _neg_words():
    with open(current_dir + '/kamus/combined_neg_words.txt', 'r') as f:
        lines = f.readlines()    
    return set([w.replace("\n", '') for w in lines])

def _pos_words():
    with open(current_dir + '/kamus/combined_pos_words.txt', 'r') as f:
        lines = f.readlines()    
    return set([w.replace("\n", '') for w in lines])

def sentiment_class(sentence):
    sentence = sentence.lower()

    if sentence in _pos_words():
        return 'pos'
    elif sentence in _neg_words():
        return 'neg'
    
    sentence_tokenize = tokenize(sentence)
    sentence_tokenize = remove_stopwords_indo(sentence_tokenize)

    sent_size = len(sentence_tokenize)
    pos_val = 0
    neg_val = 0
    net_val = 0

    pos_val = len([pos for pos in sentence_tokenize if pos in _pos_words()]) * 1.2
    neg_val = len([neg for neg in sentence_tokenize if neg in _neg_words()]) * 1.8
    net_val = (sent_size * 0.8) - (pos_val + neg_val)

    if pos_val > neg_val and pos_val > net_val:
        return 'pos'
    elif neg_val > pos_val and neg_val > net_val:
        return 'neg'
    else:
        return 'net'