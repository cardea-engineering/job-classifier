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


# global variables
RAND_SEED = 42
TRAIN_FIELD = 'job_type_name'
TRAIN_FRACTION = 0.67
JOB_TYPES = None


# prepare data
def get_dataFrame():
    df = pd.read_csv('../data/data_with_category.csv')
    type_counter = Counter(df['job_type_name'].tolist())
    category_counter = Counter(df['job_type_category_name'].tolist())
    top_10_types = {name for name, _ in type_counter.most_common(10)}
    top_5_categories = {name for name, _ in category_counter.most_common(5)}
    field_map = {
        'job_type_category_name': top_5_categories,
        'job_type_name': top_10_types
    }
    if TRAIN_FIELD in field_map:
        df = df[df[TRAIN_FIELD].map(lambda x: x in field_map[TRAIN_FIELD])]
    
    df['desc_tokenized'] = df.apply(lambda row: parse_raw_html(str(row['description']) + row['title']), axis=1)
    return df.sample(frac=1, random_state=RAND_SEED)


def set_job_types(job_types):
    global JOB_TYPES
    JOB_TYPES = job_types

def get_X_y(df):
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
    job_types, y = np.unique(df[TRAIN_FIELD], return_inverse=True)
    set_job_types(job_types)

    return X, y


def get_train_test(X, y, train_size = TRAIN_FRACTION):
    n_train = int(len(X) * train_size)
    Xtr, Xts = X[:n_train, :], X[n_train:, :]
    ytr, yts = y[:n_train], y[n_train:]
    return Xtr, Xts, ytr, yts


def get_opt_model_by_grid_search(clf, parameters, Xtr, Xts, ytr, yts):
    clf = GridSearchCV(clf, parameters)
    clf.fit(Xtr, ytr)
    model = clf.best_estimator_
    yhat = model.predict(Xts)
    print_report(yts, yhat)
    return model, yhat

def print_report(yts, yhat):
    report = classification_report(yhat, yts, labels=[i for i in range(10)], target_names=JOB_TYPES, digits=3)
    print(report)


def get_metrics_with_rules(df, Xts, yts, yhat):
    c = Counter()
    job_type_2_class_id = {job_type: i for i, job_type in enumerate(JOB_TYPES)}
    n_train = int(len(df) * TRAIN_FRACTION)

    for i, v in enumerate(Xts):
        title = df.iloc[n_train+i]['title']
        matched_type = match_job_type_rules(title.lower())
        if matched_type:
            if yhat[i] != job_type_2_class_id[matched_type]:
                c[matched_type] += 1
                yhat[i] = job_type_2_class_id[matched_type]
    
    print('prediction metrics after applying manual rules')
    print_report(yts, yhat)