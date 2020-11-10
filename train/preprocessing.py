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

# import feature engineering functions from utilities
from utils.utils import parse_raw_html
