import time
import winsound
import pandas as pd

from pilichm.main.ClassicML import run_classical_model
from pilichm.main.Constants import Constants

if __name__ == '__main__':
    print('main() - START.')
    start_time = time.time()

    # twitterApi = TwitterApiWrapper()
    # twitterApi.download_tweet_data(1_000)

    data = pd.read_csv(Constants.DATASET_FILE_PATH, delimiter='\t')
    # data = Utils.clean_data(data)
    # data = Utils.create_labels(dataset=data)

    # GaussianNB.
    # run_classical_model(dataset=data, model_name='GaussianNB', display_diagrams=True)

    # SVC.
    # run_classical_model(dataset=data, model_name='SVC', display_diagrams=True)

    # Logistic Regression.
    # run_classical_model(dataset=data, model_name='LogisticRegression', display_diagrams=True)

    data.to_csv(Constants.DATASET_FILE_PATH, sep='\t', index=False)

    print(f"Execution time: {round(time.time() - start_time, 2)} seconds")
    print(f"Execution time: {round((time.time() - start_time) / 60, 2)} minutes")
    winsound.Beep(Constants.SIGNAL_FREQ, Constants.SIGNAL_DURATION)
    print('main() - END.')
