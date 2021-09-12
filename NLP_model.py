# Import packages
import pandas as pd
from collections import Counter
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from keras.models import Sequential, load_model
from keras import callbacks
from keras.layers import Dense, Dropout
import keras_tuner as kt
import time

# READ DATA
# ISO-8859-1 is a latin encoding, with support for multiple languages,
# which uses a single byte to represent each character instead of 2 like UTF-8 does
df_train = pd.read_csv('ML_Code_test 1/train_sections_data.csv', encoding="ISO-8859-1")
df_train = df_train.iloc[:, :-3]  # remove last three columns of dataframe which are empty
df_test = pd.read_csv('ML_Code_test 1/test_sections_data.csv', encoding="ISO-8859-1")
pd.set_option('display.max_columns', None)  # don't truncate columns
print(df_train.head(5))
print(df_test.head(5))

# PRE-PROCESS DATA
# check for missing data
print(df_train.isna().sum())  # shows no null values across all columns

# check if data is balanced by printing the frequency of each label
label_0 = df_train[df_train.Label == 0].shape[0]
label_1 = df_train[df_train.Label == 1].shape[0]
print('\nlabel counts {}', format(Counter(df_train.iloc[:, 9])))
# (dealing with class-imbalance after vectorization of text-column)

# dimensionality reduction
print(df_train.FontType.unique())  # shows that this string-feature has only one value, i.e., feature can be removed
df_train = df_train.drop(['FontType'], axis=1)
df_test = df_test.drop(['FontType'], axis=1)

# encode discrete string-features numerically
ord_enc = OrdinalEncoder()
df_train["IsBold"] = ord_enc.fit_transform(df_train[["IsBold"]])  # .astype(int)
df_train["IsItalic"] = ord_enc.fit_transform(df_train[["IsItalic"]])
df_train["IsUnderlined"] = ord_enc.fit_transform(df_train[["IsUnderlined"]])
df_test["IsBold"] = ord_enc.fit_transform(df_test[["IsBold"]])
df_test["IsItalic"] = ord_enc.fit_transform(df_test[["IsItalic"]])
df_test["IsUnderlined"] = ord_enc.fit_transform(df_test[["IsUnderlined"]])

# normalize continuous features to re-scale the axis of the feature space
scaler = MinMaxScaler()
df_train.iloc[:, 4:8] = scaler.fit_transform(df_train.iloc[:, 4:8])
df_test.iloc[:, 4:8] = scaler.fit_transform(df_test.iloc[:, 4:8])

# print result
print(df_train.head(5))

# extract labels
train_y = df_train.iloc[:, 8]
test_y = df_test.iloc[:, 8]

# normalize text feature: lemmatize texts in preparation for tf-idf vectorization
wn_lemmatizer = nltk.stem.WordNetLemmatizer()


def wn_lemmatize(text):
    # lemmatize word-by-word using tokenizer
    return [wn_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)]


train_text = df_train.iloc[:, 0].apply(wn_lemmatize).apply(lambda x: ' '.join(x))
test_text = df_test.iloc[:, 0].apply(wn_lemmatize).apply(lambda x: ' '.join(x))
# print(train_text.head(5))

# vectorize texts according to bag-of-words model and turn back into dataframes
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
train_text_vectorized = tfidf_vectorizer.fit_transform(train_text).todense()  # to_dense from sparse matrix
train_text_vectorized = pd.DataFrame(train_text_vectorized, columns=tfidf_vectorizer.get_feature_names())
test_text_vectorized = tfidf_vectorizer.transform(test_text).todense()
test_text_vectorized = pd.DataFrame(test_text_vectorized, columns=tfidf_vectorizer.get_feature_names())

# concatenate vectorised texts with remaining columns into complete feature vectors
train_x = pd.concat([train_text_vectorized, df_train.iloc[:, 1:8]], axis=1)
test_x = pd.concat([test_text_vectorized, df_test.iloc[:, 1:8]], axis=1)
# checking last column after concatenation
# print(train_x.iloc[:, -1:].head(5))

# deal with heavy class-imbalance in data (over-sampling: SMOTETomek)
# smote_t = SMOTETomek(random_state=100, n_jobs=-1)
# train_x, train_y = smote_t.fit_resample(train_x, train_y)
# print('resampled labels', format(Counter(train_y)))

###############################################################
# SUPPORT VECTOR MACHINE
# Train SVM classifier with hyperparameter tuning
# svc = SVC(kernel='linear')  # define model
# parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
# grid_search = GridSearchCV(svc, parameters)
# grid_search.fit(train_x, train_y)
# print('Best accuracy:', grid_search.best_score_)
# print('Best parameters:\n', grid_search.best_params_)

# Train SVM classifier without hyperparameter tuning
# svc = SVC(kernel='linear')
# svc.fit(train_x, train_y)
# predictions = svc.predict(test_x)
# f1_score = f1_score(test_y, predictions, average='micro')  # get performance metric
# print('f1 score: ', f1_score)
# print(confusion_matrix(predictions, test_y))  # print precision/recall details
###############################################################

# NEURAL NETWORK

start_training = time.time()

global n_dim
n_dim = train_x.shape[1]  # no. of dimensions


# define model, search for best network topology architecture with given number of layers
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units1', min_value=60, max_value=120, step=20),
                        input_shape=(n_dim,), activation='relu'))
        model.add(Dropout(0.2))  # reduce overfitting via dropout regularization
        model.add(Dense(units=hp.Int('units2', min_value=40, max_value=100, step=20), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=hp.Int('units3', min_value=20, max_value=80, step=20), activation='relu'))
        model.add(Dense(units=hp.Int('units4', min_value=10, max_value=60, step=20), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))  # 1 output node, sigmoid for binary clf

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


# tuning model
tuner = kt.RandomSearch(MyHyperModel(), seed=100, objective='val_accuracy', max_trials=50)
tuner.search_space_summary()  # print hypertuning details (hyperparameter values to be explored)
# stop training after 5 consecutive epochs without improvement
stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_x, train_y, epochs=50, validation_split=0.1, callbacks=[stop_early])  # search for optimal values
net_optimised_model = tuner.get_best_models()[0]  # get model with best network topology

# tune best model's hyperparameters (epoch and batch size)
# use wrapper to enable scikit learn gridsearch on keras ann-model
k_clf = KerasClassifier(build_fn=lambda: net_optimised_model)
parameters = dict(batch_size=np.array([10, 50, 100]), epochs=np.array([25, 50, 100]))
gs = GridSearchCV(estimator=k_clf, param_grid=parameters)
gs_result = gs.fit(train_x, train_y)
hp_optimised_model = gs_result.best_estimator_  # get model with best hyperparameters

stop_training = time.time()
training_time = round(stop_training - start_training, 2)

print('Results from search for best network-topology:')
tuner.results_summary()  # print details of topology-configuration search in descending sets of hyperparameters
predictions = gs_result.predict(test_x)  # predicted labels from most optimal estimator found in grid search
print('Best parameters:\n', gs_result.best_params_)
print('Best accuracy:', accuracy_score(test_y, predictions))
print('Best f1 score:', f1_score(test_y, predictions, average='micro'))
print('Training and hypertuning time (h):', training_time)

# Save the final model with the best network topology and hyperparameters
hp_optimised_model.model.save('clf_model')  # save model architecture with trained weights

###############################################################

# # Reload optimised and trained model
# start_t = time.time()
# hp_optimised_model = load_model('clf_model')
# hp_optimised_model.summary()
# eval = hp_optimised_model.evaluate(test_x, test_y, verbose=0)
# stop_t = time.time()
# print('Accuracy:', eval[1])
# time = round(stop_t - start_t, 2)
# print('Reloading time:', time)
