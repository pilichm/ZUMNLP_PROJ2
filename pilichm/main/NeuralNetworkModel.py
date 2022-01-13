import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.utils.vis_utils import plot_model

from pilichm.main.Constants import Constants


def change_sentiment(sentiment):
    if sentiment == 'negative' or sentiment == -1:
        return 0
    elif sentiment == 'neutral' or sentiment == 0:
        return 1
    elif sentiment == 'positive' or sentiment == 1:
        return 2
    else:
        return 0


def get_model(model_name, max_words, max_len, num_of_labels):
    model = Sequential()

    # Return dense model.
    if model_name == 'dense':
        model.add(layers.Embedding(max_words, 40, input_length=max_len))
        model.add(layers.Flatten())
        model.add(layers.Dense(30, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(num_of_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Return Bidirectional LSTM model.
    elif model_name == 'lstm':
        model.add(layers.Embedding(max_words, 40, input_length=max_len))
        model.add(layers.Bidirectional(layers.LSTM(20, dropout=0.05)))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_of_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Return Bidirectional LSTM with convolution layer.
    else:
        model.add(layers.Embedding(max_words, 40, input_length=max_len))
        model.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),
                                bias_regularizer=regularizers.l2(2e-3)))
        model.add(layers.MaxPooling1D(5))
        model.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
        model.add(layers.Dense(3, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def run_neural_network_model(path_to_csv, epoch_count=25, display_diagrams=False, save_model=False, read_model=False, model_name='lstm'):

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

    max_words = 5000
    max_len = 200

    labels = tf.keras.utils.to_categorical(labels, 3, dtype="float32")

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweets = pad_sequences(sequences, maxlen=max_len)

    x_train, x_test, y_train, y_test = train_test_split(tweets, labels, random_state=0, test_size=0.3)

    print(pd.DataFrame(data=y_test).value_counts())

    if not read_model:
        model = get_model(model_name=model_name, max_words=max_words, max_len=max_len, num_of_labels=3)
    else:
        model = keras.models.load_model(Constants.PATH_TO_NN_MODEL)

    # display model
    print(model.summary())
    plot_model(model, show_shapes=True, show_layer_names=True)

    checkpoint2 = ModelCheckpoint("best_model2.hdf5",
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='auto',
                                  period=1,
                                  save_weights_only=False)
    history = model.fit(x_train,
                        y_train,
                        epochs=epoch_count,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpoint2])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    if save_model:
        model.save(Constants.PATH_TO_NN_MODEL)

    predictions = model.predict(x_test)
    print('Model accuracy: ', test_acc)

    print(f'Predictions: {np.around(predictions, decimals=0).argmax(axis=1)}')

    print(f'Actual: {y_test.argmax(axis=1)}')

    if display_diagrams:
        matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))

        print(matrix)

        conf_matrix = pd.DataFrame(matrix, index=['Neutral', 'Negative', 'Positive'],
                                   columns=['Neutral', 'Negative', 'Positive'])
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(15, 15))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})

        plt.show()
