# Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# read data
# ISO-8859-1 is a latin encoding, with support for multiple languages,
# which uses a single byte to represent each character instead of 2 like UTF-8 does
df_train = pd.read_csv('ML_Code_test 1/train_sections_data.csv', encoding="ISO-8859-1")
pd.set_option('display.max_columns', None)  # don't truncate columns
df_train = df_train.iloc[:, :-3]  # remove last three columns of dataframe which are empty
df_test = pd.read_csv('ML_Code_test 1/test_sections_data.csv', encoding="ISO-8859-1")
pd.set_option('display.max_columns', None)
print(df_train.head(5))
print(df_test.head(5))

# pre-process data
# check for missing data
print(df_train.isna().sum())  # shows no null values across all columns

# check if data is balanced by printing the frequency of each label
label_0 = df_train[df_train.Label == 0].shape[0]
label_1 = df_train[df_train.Label == 1].shape[0]
print('\nnumber of data points with label 0 =', label_0)
print('number of data points with label 1 =', label_1)
print('ratio of labels, 1/0 = ', round(label_1/label_0, 2))

# deal with heavy class-imbalance in data
###############################################################
###############################################################

# extract labels
train_y = df_train.iloc[:, 9]
test_y = df_test.iloc[:, 9]

# extract texts
train_text = df_train.iloc[:, 0]
test_text = df_test.iloc[:, 0]

# vectorize texts and turn back into dataframes
vectorizer = CountVectorizer(stop_words='english')
train_text_vectorized = vectorizer.fit_transform(train_text).todense()
train_text_vectorized = pd.DataFrame(train_text_vectorized, columns=vectorizer.get_feature_names())
test_text_vectorized = vectorizer.transform(test_text).todense()
test_text_vectorized = pd.DataFrame(test_text_vectorized, columns=vectorizer.get_feature_names())

# train model, make predictions and estimate performance
lr = LogisticRegression()
lr.fit(train_text_vectorized, train_y)
score = lr.score(test_text_vectorized, test_y)
print(score)

# NOTES
# don't perform undersampling for class-imbalance
# use cross validation to measure f1-score
