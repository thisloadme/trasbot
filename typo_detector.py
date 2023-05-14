import difflib
from nltk_utils import tokenize

def is_similar(first, second):
    return difflib.SequenceMatcher(None, first, second).ratio() > 0.7

def tokenize_correct_typo(sentence, all_words):
    words = tokenize(sentence)
    ignore_words = ['?', '!', '<', '>', '.', ',']
    for (idx, word) in enumerate(words):
        if word not in ignore_words and word not in all_words:
            valid = [w for w in all_words if is_similar(word, w)]
            words[idx] = valid[0] if len(valid) > 0 else word
    
    return words