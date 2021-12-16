from pathlib import Path
from pilichm.main.Constants import Constants
from time import sleep

import tweepy as tweepy
import pandas as pd


def get_data_from_file(filepath):
    line = None
    try:
        with open(filepath) as f:
            line = f.readline()
    except FileNotFoundError:
        print(f'No file for {filepath}.')
    finally:
        return line


# Class for downloading data from Twitter API.
class TwitterApiWrapper:

    # If no value is passed for twitter access data, app will tried to get the from text files defined in Constants.
    # Path to class: pilichm.main.Constants
    def __init__(self, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None):

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

        # Indicates whether all credentials are set.
        self.can_download_data = True

        if self.consumer_key is None:
            self.consumer_key = get_data_from_file(Constants.CONSUMER_KEY_FILE_PATH)

        if self.consumer_secret is None:
            self.consumer_secret = get_data_from_file(Constants.CONSUMER_SECRET_FILE_PATH)

        if self.access_token is None:
            self.access_token = get_data_from_file(Constants.ACCESS_TOKEN_FILE_PATH)

        if self.access_token_secret is None:
            self.access_token_secret = get_data_from_file(Constants.ACCESS_TOKEN_SECRET_FILE_PATH)

        if not self.consumer_key or not self.consumer_secret or not self.access_token or not self.access_token_secret:
            self.can_download_data = False
        else:
            # Create api from credentials.
            self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
            self.auth.set_access_token(self.access_token, self.access_token_secret)
            self.api = tweepy.API(self.auth)

        print('Twitter API created with:')
        print(f'consumer_key: {self.consumer_key}')
        print(f'consumer_secret: {self.consumer_secret}')
        print(f'access_token: {self.access_token}')
        print(f'access_token_secret: {self.access_token_secret}')
        print(f'can_download_data: {self.can_download_data}')

    # Download tweet data.
    # num_to_download - number of tweets do download.
    def download_tweet_data(self, num_to_download=1):
        datasetPath = Path(Constants.DATASET_FILE_PATH)

        # Create dataset file if not exists.
        if not datasetPath.is_file():
            with open(Constants.DATASET_FILE_PATH, 'w+') as f:
                f.write(f'{Constants.COL_TWEET_ID}\t{Constants.COL_TWEET_TEXT}\t{Constants.COL_TWEET_SENTIMENT}\n')

        # Check how many rows are already downloaded.
        input_dataset = pd.read_csv(Constants.TWEET_ID_FILE_WITH_PATH, delimiter='\t', usecols=['tweet_id'])
        output_dataset = pd.read_csv(Constants.DATASET_FILE_PATH, delimiter='\t',
                                     usecols=[Constants.COL_TWEET_ID, Constants.COL_TWEET_TEXT,
                                              Constants.COL_TWEET_SENTIMENT])

        start_index = len(output_dataset.index)
        end_index = len(input_dataset.index)
        print(f'start_index: {start_index}')

        if start_index >= end_index:
            print('All tweets already downloaded!')
            return

        used_indexes = []
        if Path(Constants.USED_IDS_FILENAME).is_file():
            with open(Constants.USED_IDS_FILENAME, 'r') as f:
                for line in f:
                    used_indexes.append(str(line.rstrip()))

        # print(used_indexes)

        indexes = []
        for i in range(start_index, end_index):
            # print(f'I: {i}')
            if num_to_download > 0:

                # Save data in packs of 10.

                # if i != 0 and i % 10 == 0:
                #     print('Saving.')
                #     output_dataset.to_csv(Constants.DATASET_FILE_PATH, sep='\t', index=False)
                #
                #     with open(Constants.USED_IDS_FILENAME, 'w') as f:
                #         for index in used_indexes:
                #             f.write(f'{index}\n')

                current_tweet_id = input_dataset.loc[i, 'tweet_id']
                # print(f'Current id: {current_tweet_id}')

                # print(f'{str(current_tweet_id)} in {used_indexes} ? {current_tweet_id in used_indexes}')

                if str(current_tweet_id) not in used_indexes:
                    indexes.append(current_tweet_id)
                    used_indexes.append(current_tweet_id)
                    # print('APPENDING')
                    # print(f'APPENDED LEN {len(indexes)}')
                    num_to_download -= 1
                    if len(indexes) == 100:
                        try:
                            # used_indexes.append(current_tweet_id)
                            # tweet = self.api.get_status(current_tweet_id)
                            tweet_list = self.api.lookup_statuses(indexes)
                            # print(f'Call res: {tweet_list}')
                            for tweet in tweet_list:
                                tweet_text = tweet.text.rstrip()
                                tweet_text = tweet_text.replace('\n', ' ')
                                print(f'Current tweet text: {tweet_text}')

                                new_data = {
                                    Constants.COL_TWEET_ID: tweet.id,
                                    Constants.COL_TWEET_TEXT: tweet_text,
                                    Constants.COL_TWEET_SENTIMENT: 'brak'
                                }

                                print(new_data)
                                output_dataset = output_dataset.append(new_data, ignore_index=True)
                                output_dataset.to_csv(Constants.DATASET_FILE_PATH, sep='\t', index=False)

                            with open(Constants.USED_IDS_FILENAME, 'w') as f:
                                for index in used_indexes:
                                    f.write(f'{index}\n')
                                # break

                        except tweepy.errors.NotFound:
                            print(f'No tweet found for id: {current_tweet_id}')
                        except tweepy.errors.Forbidden:
                            print(f'User suspended for tweet of id: {current_tweet_id}')
                        except tweepy.errors.TooManyRequests:
                            print('429 Too Many Requests')
                            print('Saving.')
                            output_dataset.to_csv(Constants.DATASET_FILE_PATH, sep='\t', index=False)

                            with open(Constants.USED_IDS_FILENAME, 'w') as f:
                                for index in used_indexes:
                                    f.write(f'{index}\n')
                            break

                        sleep(0.5)
                        indexes.clear()
            else:
                print('Downloaded required amount - saving.')
                output_dataset.to_csv(Constants.DATASET_FILE_PATH, sep='\t', index=False)

                with open(Constants.USED_IDS_FILENAME, 'w') as f:
                    for index in used_indexes:
                        f.write(f'{index}\n')

                break
