# pre-processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from utils.utils import parse_raw_html
from utils.utils import STOP_WORDS

# model training and evaluation
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from utils.match_rules import match_job_type_rules

# serialize models
import pickle


df['desc_tokenized'] = df.apply(lambda row: parse_raw_html(str(row['description']) + row['title']), axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# prepare data
def get_X_y(df):

    type_counter = Counter(df['job_type_name'].tolist())
    category_counter = Counter(df['job_type_category_name'].tolist())
    top_10_types = {name for name, _ in type_counter.most_common(10)}
    top_5_categories = {name for name, _ in category_counter.most_common(5)}
    pred_field_name = 'job_type_name'
    field_map = {
        'job_type_category_name': top_5_categories,
        'job_type_name': top_10_types
    }
    if pred_field_name in field_map:
        df = df[df[pred_field_name].map(lambda x: x in field_map[pred_field_name])]

    tfIdfVectorizer = TfidfVectorizer(
        analyzer='word', 
        sublinear_tf=True,
        stop_words=STOP_WORDS,
        strip_accents='unicode',
        token_pattern=r'\w{2,}',
        ngram_range=(1,1),
        max_features=1000,
        use_idf=True
    )
    tfIdf = tfIdfVectorizer.fit_transform(list(df['desc_tokenized']))
    X = tfIdf.toarray() # convert to dense array
    job_types, y = np.unique(df[pred_field_name], return_inverse=True)
    return X, y, job_types


def get_train_test(X, y, train_size = 0.67):
    n_samples = len(X)
    n_train = int(n_samples * train_size)
    Xtr, Xts = X[:n_train, :], X[n_train:, :]
    ytr, yts = y[:n_train], y[n_train:]
    return Xtr, Xts, ytr, yts


def get_opt_model_by_grid_search(clf, parameters, Xtr, Xts, ytr, yts, job_types):
    clf = GridSearchCV(clf, parameters)
    clf.fit(Xtr, ytr)
    model = clf.best_estimator_
    yhat = model.predict(Xts)
    report = classification_report(yhat, yts, labels=[i for i in range(10)], target_names=job_types, digits=3)
    print('prediction metrics without manual rules')
    print(report)
    return model

def get_result(model, Xts, yts, job_types):
    yhat = model.predict(Xts)
    report = classification_report(yhat, yts, labels=[i for i in range(10)], target_names=job_types, digits=3)
    print('prediction metrics without manual rules')
    print(report)
