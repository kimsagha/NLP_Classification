# Import packages
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# read data
# ISO-8859-1 is a latin encoding, with support for multiple languages,
# which uses a single byte to represent each character instead of 2 like UTF-8 does
df_train = pd.read_csv('ML_Code_test 1/train_sections_data.csv', encoding="ISO-8859-1")
df_train = df_train.iloc[:, :-3]  # remove last three columns of dataframe which are empty
df_test = pd.read_csv('ML_Code_test 1/test_sections_data.csv', encoding="ISO-8859-1")
pd.set_option('display.max_columns', None)  # don't truncate columns
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

# deal with heavy class-imbalance in data (over-sampling?)
###############################################################
###############################################################

# encode discrete string-features numerically
ord_enc = OrdinalEncoder()
df_train["IsBold"] = ord_enc.fit_transform(df_train[["IsBold"]]).astype(int)
df_train["IsItalic"] = ord_enc.fit_transform(df_train[["IsItalic"]]).astype(int)
df_train["IsUnderlined"] = ord_enc.fit_transform(df_train[["IsUnderlined"]]).astype(int)
df_test["IsBold"] = ord_enc.fit_transform(df_test[["IsBold"]]).astype(int)
df_test["IsItalic"] = ord_enc.fit_transform(df_test[["IsItalic"]]).astype(int)
df_test["IsUnderlined"] = ord_enc.fit_transform(df_test[["IsUnderlined"]]).astype(int)

# dimensionality reduction
print(df_train.FontType.unique())  # shows that this string-feature has only one value, i.e., feature can be removed
df_train = df_train.drop(['FontType'], axis=1)
df_test = df_test.drop(['FontType'], axis=1)

# print result
print(df_train.head(5))

# extract labels
train_y = df_train.iloc[:, 8]
test_y = df_test.iloc[:, 8]

# extract texts
train_text = df_train.iloc[:, 0]
test_text = df_test.iloc[:, 0]

# vectorize texts and turn back into dataframes
vectorizer = TfidfVectorizer(stop_words='english')
train_text_vectorized = vectorizer.fit_transform(train_text).todense()
train_text_vectorized = pd.DataFrame(train_text_vectorized, columns=vectorizer.get_feature_names())
test_text_vectorized = vectorizer.transform(test_text).todense()
test_text_vectorized = pd.DataFrame(test_text_vectorized, columns=vectorizer.get_feature_names())

# concatenate vectorised texts with remaining columns into complete feature vectors
train_x = pd.concat([train_text_vectorized, df_train.iloc[:, 1:8]], axis=1)
test_x = pd.concat([test_text_vectorized, df_test.iloc[:, 1:8]], axis=1)

print('check nulls in labels')
print(train_x.isna().sum())

# train model, make predictions and estimate performance
lr = LogisticRegression(max_iter=3000)  # class_weight='balanced'
lr.fit(train_x, train_y)
predictions = lr.predict(test_x)
f1_score = f1_score(test_y, predictions, average='micro')
print('f1 score: ', f1_score)
print(confusion_matrix(predictions, test_y))

# NOTES
# normalize left, right, top and bottom columns?
# how much pre-processing should be done, w.r.t. lemmatizer, stemmatizer, stopwords, etc.
# don't perform undersampling for class-imbalance
# use cross validation to measure f1-score
