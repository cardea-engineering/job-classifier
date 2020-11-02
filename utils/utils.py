import re
import html
import unicodedata
import pickle
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.match_rules import match_job_type_rules


def get_model(file_name):
    return pickle.load(open('models/' + file_name, 'rb'))

with open('static/stopwords.txt') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]

job_category_model = get_model('job_type_category_name_model.pkl')
job_type_model = get_model('job_type_name_model.pkl')
job_exp_model = get_model('job_exp_name_model.pkl')
tfidfVectorizer = get_model('tfIdfVectorizer.pkl')

job_categories = ['Business Operations', 'Data & Analytics',
                  'Finance, Legal & Compliance', 'Product & Design',
                  'Software Engineering']

job_types = ['Back-End Software Engineering', 'Business Development',
             'Business Intelligence & Data Analysis', 'Data Engineering',
             'DevOps & Infrastructure', 'Front-End Software Engineering',
             'Full-Stack Software Engineering', 'Operations & General Business',
             'Product Manager', 'Sales']


job_experiences = ['Entry Level (New Grad)', 'Intern', 'Junior (1-2 years)']

lemmatizer = WordNetLemmatizer()


def tokenize_text(sentence):
    token_words = word_tokenize(sentence)
    return " ".join([lemmatizer.lemmatize(word.lower(), pos='v') for word in token_words])


def parse_raw_html(raw_html):
    if isinstance(raw_html, str):
        soup = BeautifulSoup(html.unescape(raw_html), features="html.parser")
        return tokenize_text(unicodedata.normalize("NFKD", soup.get_text()))
    return ""


def wrap_result(names, probs):
    return sorted([
        {'name': names[i], 'probability': float('%.3f' % p)}
        for i, p in enumerate(probs)], key=lambda x: -x['probability']
    )


def predict(job_title, job_desc):
    text_input = parse_raw_html(job_title + job_desc)
    Xts = tfidfVectorizer.transform(np.array([text_input])).toarray()

    type_matched = match_job_type_rules(job_title, job_desc)
    if type_matched:
        result_type = wrap_result([type_matched], [1])
    else:
        prob_types = job_type_model.predict_proba(Xts).flatten()
        result_type = wrap_result(job_types, prob_types)

    prob_category = job_category_model.predict_proba(Xts).flatten()
    result_category = wrap_result(job_categories, prob_category)
    prob_exp = job_exp_model.predict_proba(Xts).flatten()
    result_exp = wrap_result(job_experiences, prob_exp)

    return {
        'job_category': result_category,
        'job_type': result_type,
        'job_experience': result_exp
    }
