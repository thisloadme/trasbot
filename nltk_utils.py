import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def sentence_tokenize(paragraph):
    return nltk.sent_tokenize(paragraph)

def stem(word):
    return stemmer.stem(word.lower())

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