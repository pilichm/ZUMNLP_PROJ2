import time
import winsound
import pandas as pd
from tensorflow.python.client import device_lib
import tensorflow as tf
from pilichm.main import Utils
# from pilichm.main.BERTModel import run_bert_model
from pilichm.main.ClassicML import run_classical_model
from pilichm.main.Constants import Constants
from pilichm.main.NeuralNetworkModel import run_neural_network_model
from pilichm.main.TwitterApiWrapper import TwitterApiWrapper
from pilichm.main.Utils import display_sentiment_distribution

if __name__ == '__main__':
    print('main() - START.')
    start_time = time.time()

    try:
        twitterApi = TwitterApiWrapper()
        twitterApi.download_tweet_data(20_000)

        # data = pd.read_csv(Constants.DATASET_FILE_PATH, delimiter='\t')
        # data['CLEANED_TEXT'] = '_'
        # Utils.clean_data(path_to_csv=Constants.DATASET_FILE_PATH)
        # Utils.create_labels(path_to_csv=Constants.DATASET_FILE_PATH)

        # print('Sentiment count:')
        # print(data['sentiment'].value_counts())
        # display_sentiment_distribution(path_to_csv=Constants.DATASET_FILE_PATH)
        # GaussianNB.
        # run_classical_model(path_to_csv=Constants.DATASET_FILE_PATH, model_name='GaussianNB', display_diagrams=True)

        # SVC.
        # run_classical_model(path_to_csv=Constants.DATASET_FILE_PATH, model_name='SVC', display_diagrams=True)

        # Logistic Regression.
        # run_classical_model(path_to_csv=Constants.DATASET_FILE_PATH, model_name='LogisticRegression', display_diagrams=True)

        # Neural Network Model.

        # data.to_csv(Constants.DATASET_FILE_PATH, sep='\t', index=False)
        # run_neural_network_model(path_to_csv=Constants.DATASET_FILE_PATH, display_diagrams=True, read_model=False,
        #                          save_model=True, epoch_count=25)

        # Run BERT model.
        # run_bert_model(path_to_csv=Constants.DATASET_FILE_PATH, display_diagrams=True, read_model=False, save_model=True
        #                , batch_size=16, epoch_count=5, train_model=True)
    finally:
        print(f"Execution time: {round(time.time() - start_time, 2)} seconds")
        print(f"Execution time: {round((time.time() - start_time) / 60, 2)} minutes")
        winsound.Beep(Constants.SIGNAL_FREQ, Constants.SIGNAL_DURATION)
        print('main() - END.')
