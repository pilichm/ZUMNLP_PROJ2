import math

import numpy as np
import pandas as pd
import seaborn as sns
import random
from itertools import zip_longest
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

from pilichm.main.Constants import Constants


def create_dataset(features, labels, batch_size):
    features = np.squeeze(features)
    features = np.array(list(zip_longest(*features, fillvalue=0))).T
    return tf.data.Dataset.from_tensor_slices((tf.cast(features, tf.int32), tf.cast(labels, tf.int32)))\
        .batch(batch_size)


def run_bert_model(path_to_csv, display_diagrams=False, batch_size=128, epoch_count=25, save_model=False,
                   read_model=False, train_model=False):
    data = pd.read_csv(path_to_csv, delimiter='\t',
                       usecols=[Constants.COL_CLEANED_TEXT, Constants.COL_TWEET_SENTIMENT.lower()]).head(10_000)

    if not read_model:
        # Download pre trained BERT model.
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    else:
        # Read saved model.
        model = keras.models.load_model(Constants.PATH_TO_BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # Tokenize tweets text.
    data[Constants.COL_CLEANED_TEXT] = data[Constants.COL_CLEANED_TEXT] \
        .apply(lambda row: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row)))

    data[Constants.COL_TWEET_SENTIMENT.lower()] = data[Constants.COL_TWEET_SENTIMENT.lower()]\
        .apply(lambda element: element + 1)

    # Split data into train, test and validation. The create datasets.
    tweets_array = np.array(data[Constants.COL_CLEANED_TEXT])
    sentiment_array = np.array(data[Constants.COL_TWEET_SENTIMENT.lower()])

    tweets_train, tweets_test, sentiment_train, sentiment_test = \
        train_test_split(tweets_array, sentiment_array, random_state=0, test_size=0.3)

    tweets_train, tweets_validation, sentiment_train, sentiment_validation = \
        train_test_split(tweets_train, sentiment_train, random_state=0, test_size=0.1)

    number_of_labels = len(np.unique(sentiment_validation))

    print(f'Train dataset size: {len(tweets_train)}')
    print(f'Validationdata set size: {len(tweets_validation)}')
    print(f'Test dataset size: {len(tweets_validation)}')
    print(f'Number of labels: {number_of_labels}')

    # Compile model.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    tweets_train = np.squeeze(tweets_train)
    tweets_validation = np.squeeze(tweets_validation)
    tweets_test = np.squeeze(tweets_test)

    tweets_train = np.array(list(zip_longest(*tweets_train, fillvalue=0))).T
    tweets_validation = np.array(list(zip_longest(*tweets_validation, fillvalue=0))).T
    tweets_test = np.array(list(zip_longest(*tweets_test, fillvalue=0))).T

    print(f'Features shape {tweets_train.shape}')
    print(f'Labels shape {sentiment_train.shape}')

    train_dataset = create_dataset(tweets_train, sentiment_train, batch_size)
    validation_dataset = create_dataset(tweets_validation, sentiment_validation, batch_size)
    test_dataset = create_dataset(tweets_test, sentiment_test, batch_size)

    # train_dataset = train_dataset.batch(batch_size)
    # test_dataset = test_dataset.batch(batch_size)
    # validation_dataset = validation_dataset.batch(batch_size)

    if train_model:
        model.fit(train_dataset, validation_data=validation_dataset, epochs=epoch_count)

    if save_model:
        model.save(Constants.PATH_TO_BERT_MODEL)

    predictions = model.predict(test_dataset)
    predictions = np.argmax(predictions.logits, axis=1)

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Model test accuracy: {test_acc}')

    print(classification_report(y_true=sentiment_test, y_pred=predictions))

    if display_diagrams:
        matrix = confusion_matrix(sentiment_test, predictions)

        print(matrix)

        conf_matrix = pd.DataFrame(matrix, index=['Neutral', 'Negative', 'Positive'],
                                   columns=['Neutral', 'Negative', 'Positive'])
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(15, 15))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})

        plt.show()
