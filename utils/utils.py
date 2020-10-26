import re
import html
import unicodedata
import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


job_categories = ['Business Operations', 'Data & Analytics',
                  'Finance, Legal & Compliance', 'Product & Design',
                  'Software Engineering']

job_types = ['Back-End Software Engineering', 'Business Development',
             'Business Intelligence & Data Analysis', 'Data Engineering',
             'DevOps & Infrastructure', 'Front-End Software Engineering',
             'Full-Stack Software Engineering', 'Operations & General Business',
             'Product Manager', 'Sales']

lemmatizer = WordNetLemmatizer()


def tokenize_text(sentence):
    token_words = word_tokenize(sentence)
    tokens = []
    for word in token_words:
        if re.match('[a-zA-Z]{2,}', word) and word not in ENGLISH_STOP_WORDS:
            tokens.append(lemmatizer.lemmatize(word.lower(), pos='v'))
    return " ".join(tokens)


def parse_raw_html(raw_html):
    if isinstance(raw_html, str):
        soup = BeautifulSoup(html.unescape(raw_html), features="html.parser")
        return tokenize_text(unicodedata.normalize("NFKD", soup.get_text()))
    return ""


def get_serialized(file_name):
    return pickle.load(open('static/' + file_name, 'rb'))
