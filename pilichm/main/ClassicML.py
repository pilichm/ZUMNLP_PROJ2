from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
import sklearn.svm

from pilichm.main import Utils
from pilichm.main.Constants import Constants


def get_class(row):
    data = row.values
    if data[0] > data[1] and data[0] > data[2]:
        return -1
    elif data[1] > data[0] and data[1] > data[2]:
        return 0
    elif data[2] > data[0] and data[2] > data[1]:
        return 1
    else:
        return 0


# Split dataset and prepare train, test and validation.
def prepare_data(dataset):
    print(dataset.columns)
    # data = dataset[[Constants.COL_CLEANED_TEXT, Constants.COL_TWEET_SENTIMENT.lower()]]

    # Create train, test and validation datasets.
    x = dataset[Constants.COL_CLEANED_TEXT]
    y = dataset[Constants.COL_TWEET_SENTIMENT.lower()]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=11)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=11)

    # Convert tweets to tf-idf vectors. # 500000
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4_000)
    x_train = vectorizer.fit_transform(x_train.values.astype('U'))
    x_test = vectorizer.transform(x_test.values.astype('U'))

    vocabulary = vectorizer.get_feature_names()

    x_train = pd.DataFrame(data=x_train.toarray(), columns=vocabulary).iloc[:, 0::2]
    x_test = pd.DataFrame(data=x_test.toarray(), columns=vocabulary).iloc[:, 0::2]

    return x_train, x_test, x_validation, y_train, y_test, y_validation


# Performs classical ml model of submitted name on data.
# Also displays roc and confusion matrix.
# Available models:
# 'GaussianNB', 'SVC', 'LogisticRegression'.
def run_classical_model(path_to_csv, model_name, display_diagrams=False):
    max_words = 5000
    max_len = 200

    # First model: GaussianNB.
    if model_name == 'GaussianNB':
        clf = GaussianNB()
    # Second model: Support Vector Machine, with scaled data and linear kernel.
    elif model_name == 'SVC':
        clf = make_pipeline(StandardScaler(), sklearn.svm.SVC(kernel='linear', probability=True))
    # Third model: Logistic Regression.
    elif model_name == 'LogisticRegression':
        clf = LogisticRegression(random_state=11)
    else:
        print(f'Unknown model: >{model_name}<!')
        return

    data_list = []
    labels = []
    for chunk in pd.read_csv(path_to_csv, delimiter='\t', chunksize=2_000,
                             usecols=[Constants.COL_CLEANED_TEXT, Constants.COL_TWEET_SENTIMENT.lower()]):
        data_to_list = chunk[Constants.COL_CLEANED_TEXT].values.tolist()
        labels_to_list = chunk[Constants.COL_TWEET_SENTIMENT.lower()].values.tolist()
        for data, label in zip(data_to_list, labels_to_list):
            data_list.append(data)
            labels.append(label)

    data = np.array(data_list)
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweets = pad_sequences(sequences, maxlen=max_len)

    labels = np.array(labels)
    # labels = tf.keras.utils.to_categorical(labels, 3, dtype="float32")

    x_train, x_test, y_train, y_test = train_test_split(tweets, labels, random_state=0, stratify=labels, test_size=0.3)

    clf.fit(x_train, y_train)
    # print(f'CLASSES: {clf.classes_}')
    predicted_prob = clf.predict_proba(x_test)
    # print(f'Preds; {predicted_prob}')
    df = pd.DataFrame(data=predicted_prob, columns=[clf.classes_])
    # print(f'Preds; {df.head()}')
    df['prediction'] = df.apply(get_class, axis=1)
    df.columns = ['neg', 'neut', 'pos', 'prediction']

    # Display confusion matrix and roc curves for model
    if display_diagrams:
        plot_roc_for_three_labels(y_test, df, ['neg', 'neut', 'pos'])
        plot_conf_matrix(y_test, df['prediction'], clf.classes_, ['neg', 'neut', 'pos'])


# Displays roc curves for three classes.
def plot_roc_for_three_labels(y_true, y_predicted, labels):
    fpr = {}
    tpr = {}
    thresh = {}
    colors = ['orange', 'green', 'blue']

    y_true = pd.DataFrame(y_true)
    y_true.columns = ['original']
    y_true[labels[0]] = y_true['original'].apply(lambda value: 1. if value == -1 else 0.)
    y_true[labels[1]] = y_true['original'].apply(lambda value: 1. if value == 0 else 0.)
    y_true[labels[2]] = y_true['original'].apply(lambda value: 1. if value == 1 else 0.)

    for index, label in zip(range(3), labels):
        fpr[index], tpr[index], thresh[index] = roc_curve(y_true[label], y_predicted[label])

    # Display number of matched/unmatched values for each class.
    for label in labels:
        print(f'Label {label} distribution:')
        print(f"Actual: {Counter(y_true['pos'])}")
        print(f"Predicted: {Counter(y_predicted['pos'])}")

    for index, label in zip(range(3), labels):
        plt.plot(fpr[index], tpr[index], linestyle='--', color=colors[index], label=f'{label} vs Rest')

    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC', dpi=300)

    plt.show()


# Displays confusion matrix.
def plot_conf_matrix(y_true, y_predicted, labels, displayed_labels):
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=labels)
    cf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=displayed_labels)
    cf_matrix_display.plot()
    plt.show()
