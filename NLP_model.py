# Import packages
import pandas as pd

global df_train, df_test


# Read in data
def read_data():
    global df_train, df_test
    # ISO-8859-1 is a latin encoding, with support for multiple languages,
    # which uses a single byte to represent each character instead of 2 like UTF-8 does
    df_train = pd.read_csv('ML_Code_test 1/train_sections_data.csv', encoding="ISO-8859-1")
    pd.set_option('display.max_columns', None)  # don't truncate columns
    df_train = df_train.iloc[:, :-3]  # remove last three columns of dataframe which are empty
    df_test = pd.read_csv('ML_Code_test 1/test_sections_data.csv', encoding="ISO-8859-1")
    pd.set_option('display.max_columns', None)
    print(df_train.head(5))
    print(df_test.head(5))


# Data pre-processing
def check_data():
    # check for missing data
    print(df_train.isna().sum())  # shows no null values across all columns

    # check if data is balanced by printing the frequency of each label
    label_0 = df_train[df_train.Label == 0].shape[0]
    label_1 = df_train[df_train.Label == 1].shape[0]
    print('\nnumber of data points with label 0 =', label_0)
    print('number of data points with label 1 =', label_1)
    print('ratio of labels, 1/0 = ', round(label_1/label_0, 2))

    # deal with heavy class-imbalance in data using SMOTE (Synthetic Minority Oversampling Technique)
    # SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples
    # in the feature space and drawing a new sample at a point along that line. Specifically, a random example from
    # the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5).
    # A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between
    # the two examples in feature space.
    # --> plausible new points in the feature space


read_data()
check_data()

# don't perform undersampling for class-imbalance
# use cross validation to measure f1-score
