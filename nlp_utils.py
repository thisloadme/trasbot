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
    with open(current_dir + '/combined_stop_words.txt', 'r') as f:
        lines = f.readlines()

    stopwords_indo = [w.replace("\n", '') for w in lines]
    return [w for w in tokenized_sentence if w not in stopwords_indo]

def slang_word_meaning(word):
    with open(current_dir + '/combined_slang_words.txt', 'r') as f:
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
