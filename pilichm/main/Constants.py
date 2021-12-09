from rootDir import ROOT_DIR


class Constants:
    DATA_DIR_NAME = 'data'

    # Path to file containing ids of twits for download.
    TWEET_ID_FILENAME = 'pl_covid_tweets_clean.txt'
    TWEET_ID_FILE_WITH_PATH = f'{ROOT_DIR}/{DATA_DIR_NAME}/{TWEET_ID_FILENAME}'

    # Paths to files containing twitter api credentials.
    TWITTER_API_CRED_FILE_PATH = 'D:\\t_creds\\'
    CONSUMER_KEY_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}consumer_key.txt'
    CONSUMER_SECRET_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}consumer_secret.txt'
    ACCESS_TOKEN_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}access_token.txt'
    ACCESS_TOKEN_SECRET_FILE_PATH = f'{TWITTER_API_CRED_FILE_PATH}access_token_secret.txt'

    # Path to created file containing twitter id, text and sentiment.
    DATASET_FILE_PATH = f'{ROOT_DIR}/{DATA_DIR_NAME}/twitter.csv'
    COL_TWEET_ID = 'TWEET_ID'
    COL_TWEET_TEXT = 'TEXT'
    COL_TWEET_SENTIMENT = 'SENTIMENT'

    USED_IDS_FILENAME = f'{ROOT_DIR}/{DATA_DIR_NAME}/used_indexes.txt'

    # Regular expressions for removing unnecessary elements.
    REGEX_URL = r'https?://\S+|www\.\S+'
    REGEX_EMAIL = '\S*@\S*\s?'
    REGEX_POLISH_CHARS = f'[^qwertyuioplkjhgfdsazxcvbnmęóąśżźćńQWERTYUIOPLKJHGFDSAZXCVBNMĘÓĄŚŻŹĆŃ ]'

    SIGNAL_DURATION = 500
    SIGNAL_FREQ = 440



