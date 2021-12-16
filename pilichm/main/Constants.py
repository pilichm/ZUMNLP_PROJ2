# from rootDir import ROOT_DIRT_DIR
from pilichm.main import Utils
from IPython import get_ipython


class Constants:
    if 'google.colab' in str(get_ipython()):
        ROOT_DIRT_DIR = '/content/sample_data/ZUMNLP_PROJ2/'
    else:
        ROOT_DIRT_DIR = 'C:\\Users\\Michał Pilichowski\\PycharmProjects\\ZUMNLP_PROJ2\\'

    DATA_DIR_NAME = 'data'

    # Path to file containing ids of twits for download.
    TWEET_ID_FILENAME = 'pl_covid_tweets_clean.txt'
    TWEET_ID_FILE_WITH_PATH = f'{ROOT_DIRT_DIR}/{DATA_DIR_NAME}/{TWEET_ID_FILENAME}'

    # Paths to files containing twitter api credentials.
    TWITTER_API_CRED_FILE_PATH = 'D:\\t_creds\\'
    CONSUMER_KEY_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}consumer_key.txt'
    CONSUMER_SECRET_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}consumer_secret.txt'
    ACCESS_TOKEN_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}access_token.txt'
    ACCESS_TOKEN_SECRET_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}access_token_secret.txt'

    # Path to created file containing twitter id, text and sentiment.
    DATASET_FILE_PATH = f'{ROOT_DIRT_DIR}/{DATA_DIR_NAME}/twitter.csv'
    COL_TWEET_ID = 'TWEET_ID'
    COL_TWEET_TEXT = 'TEXT'
    COL_CLEANED_TEXT = 'CLEANED_TEXT'
    COL_TWEET_SENTIMENT = 'SENTIMENT'

    USED_IDS_FILENAME = f'{ROOT_DIRT_DIR}/{DATA_DIR_NAME}/used_indexes.txt'

    # Regular expressions for removing unnecessary elements.
    REGEX_URL = r'https?://\S+|www\.\S+'
    REGEX_EMAIL = '\S*@\S*\s?'
    REGEX_POLISH_CHARS = f'[^qwertyuioplkjhgfdsazxcvbnmęóąśżźćńQWERTYUIOPLKJHGFDSAZXCVBNMĘÓĄŚŻŹĆŃ ]'

    SIGNAL_DURATION = 500
    SIGNAL_FREQ = 440

    PATH_TO_NN_MODEL = f'{ROOT_DIRT_DIR}/{DATA_DIR_NAME}/model.md'
