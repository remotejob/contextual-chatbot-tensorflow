# things we need for NLP
import nltk
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()

from nltk.stem import SnowballStemmer
print(" ".join(SnowballStemmer.languages))
stemmer = SnowballStemmer("finnish")

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)   