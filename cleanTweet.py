
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import numpy as np
import re
from spellchecker import SpellChecker
from textblob import TextBlob
from translate import Translator

#Import Training and Testing Data
#train = pd.read_csv('train.csv')
#print("Training Set:"% train.columns, train.shape, len(train))
#test = pd.read_csv('test_tweets.csv')
#print("Test Set:"% test.columns, test.shape, len(test))#Tokenize words in order to clean and stem


tok = WordPunctTokenizer()
# patterns to remove html tags numbers and special Characters
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
porter=PorterStemmer()

def translate(sentence, lenguage):
    #lenguage ha de ser un str rollo 'es' o 'en'
    
    translator = Translator(lenguage)
    
    return translator.translate(sentence)

def correct_sentence(sentence):
    spell = SpellChecker()
    misspelled = spell.unknown(sentence.split())
    correct = sentence.split()
    for word in misspelled:
        w = spell.correction(word)
        index = correct.index(word)
        correct[index] = w
    correct = [nw + ' ' for nw in correct]
    return ''.join(correct)[:-1]

def tweet_cleaner(text):
    
    
    try:
    
        punctation = [';', ',', '.', '!', '\n']
    
        for i in punctation:
            
            text = text.replace(i, '')
        
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        stripped = re.sub(combined_pat, '', souped)
        try:
            clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            clean = stripped
        
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        lower_case = letters_only.lower()
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # Tokenize and join together to remove unnecessary white spaces
        words = tok.tokenize(lower_case)
        #Stemming
        stem_sentence=[]
        for word in words:
            if not '@' == word[0]:
                stem_sentence.append(porter.stem(word))
                stem_sentence.append(" ")
        #Rejoin the words back to create the cleaned tweet
        
        words="".join(stem_sentence).strip()
        
        words = words.lower()
        #if TextBlob(words).detect_language() != 'en':
            
        words = translate(words, 'en')
               
        words = correct_sentence(words)
        
        return words
            
    except Exception as e:
        
        print(e)
        
        return None