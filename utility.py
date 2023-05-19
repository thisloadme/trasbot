import difflib
from nlp_utils import tokenize, slang_word_meaning

def is_similar(first, second):
    return difflib.SequenceMatcher(None, first, second).ratio() > 0.7

def tokenize_correct_typo_slang(sentence, all_words):
    ignore_words = ['?', '!', '<', '>', '.', ',']
    words = tokenize(sentence)
    words = [slang_word_meaning(w) for w in words if w not in ignore_words]
    for (idx, word) in enumerate(words):
        if word not in all_words:
            valid = [w for w in all_words if is_similar(word, w)]
            words[idx] = valid[0] if len(valid) > 0 else word
    
    return words