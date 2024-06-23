# preprocess.py

from string import punctuation
import nltk
from nltk import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import os
import warnings

warnings.filterwarnings("ignore")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

mystop_words = set(stopwords.words("english"))
punctuation_set = set(punctuation)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocess_books(path):
    books = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            with open(filepath, 'r', errors='ignore') as file:
                corpus = file.read()
            sent_tokens = sent_tokenize(corpus)
            for sent in sent_tokens:
                # Remove punctuation, tokenize, remove stop words, and lemmatize
                words = [lemmatizer.lemmatize(word.lower()) for word in simple_preprocess(
                    sent) if word not in mystop_words and word not in punctuation_set]
                books.append(words)
    return books


print("Done processing files.")
