import re

import numpy as np
import pandas as pd
import spacy
from IPython import get_ipython
from nltk import PorterStemmer
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.cluster import KMeans

from pilichm.main.Constants import Constants


# Removes stop words from given line. Also removes leading and trailing whitespaces.
def remove_stop_and_single_words_and_stem(line, nlp):
    new_line = ""

    for wrd in line.split():
        if wrd not in nlp.Defaults.stop_words and len(wrd) > 1:
            new_line += wrd.strip().lower()
            new_line += " "

    return new_line


# Removes repetitions pf words in submitted line of text.
def remove_repetitions(line):
    unique_words = []
    for wrd in line.split():
        if wrd not in unique_words:
            unique_words.append(wrd)

    new_line = ''
    for wrd in unique_words:
        new_line += wrd
        new_line += ' '

    return new_line


# Stem words in line.
def stem_line(line):
    stemmer = PorterStemmer()
    new_line = ''

    for wrd in line.split():
        new_line += stemmer.stem(wrd)
        new_line += " "

    return new_line


# Remove urls, emails, punctuation, emo, repetitions and apostrophes.
def clean_data(data):
    url_pattern = re.compile(Constants.REGEX_URL)
    letters_pattern = re.compile(Constants.REGEX_POLISH_CHARS)
    nlp = spacy.load('pl_core_news_sm')

    # Remove urls.
    data[Constants.COL_TWEET_TEXT] = data[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: url_pattern.sub(r' ', text))

    # Remove emails.
    data[Constants.COL_TWEET_TEXT] = data[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: re.sub(Constants.REGEX_EMAIL, ' ', text))

    # Remove all characters that aren't polish letters.
    data[Constants.COL_TWEET_TEXT] = data[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: letters_pattern.sub(r'', text))

    # Remove stop words.
    data[Constants.COL_TWEET_TEXT] = data[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: remove_stop_and_single_words_and_stem(text, nlp))

    # Remove repetitions.
    data[Constants.COL_TWEET_TEXT] = data[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: remove_repetitions(text))

    # Tokenize - stemming.
    data[Constants.COL_TWEET_TEXT] = data[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: stem_line(text))

    return data


def get_sentiments(text, words_dict):
    # print(f'>{text}<')

    total = 0
    count = 0
    for t in text:
        # print(t)
        if words_dict.get(t):
            # print(f'MATCH {words_dict.get(t)} for {t}')
            total += int(words_dict.get(t))
        count += 1

    avg = total / count

    # print(f'{avg} = {total} / {count}')

    sentiment = -1 if avg < -0.15 else 1 if avg > 0.15 else 0
    return sentiment


# Add sentiment labels to tweets in dataset.
def create_labels(dataset, num_of_classes=3):
    # Create word embeddings.
    sent = [row for row in dataset[Constants.COL_TWEET_TEXT]]
    phrases = Phrases(sent, min_count=1, progress_per=50000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    w2v_model = Word2Vec(min_count=4, window=5, sample=1e-5, alpha=0.03, min_alpha=0.0007,
                         negative=20, seed=42, workers=multiprocessing.cpu_count() - 1)

    w2v_model.build_vocab(sentences, progress_per=50000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save("word2vec.model")

    word_vectors = Word2Vec.load("word2vec.model").wv
    model = KMeans(n_clusters=num_of_classes, max_iter=1000, random_state=42, n_init=50).fit(
        X=word_vectors.vectors.astype('double'))
    word_vectors.similar_by_vector(model.cluster_centers_[1], topn=200, restrict_vocab=None)

    positive_cluster_center = model.cluster_centers_[2]
    negative_cluster_center = model.cluster_centers_[1]
    neutral_cluster_center = model.cluster_centers_[0]

    words = pd.DataFrame(word_vectors.index_to_key)
    words.columns = ['words']
    words['vectors'] = words.words.apply(lambda x: word_vectors[f'{x}'])
    words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
    words.cluster = words.cluster.apply(lambda x: x[0])

    words['cluster_value'] = [1 if i == 2 else 0 if i == 0 else -1 for i in words.cluster]
    words['closeness_score'] = words.apply(lambda x: 1 / (model.transform([x.vectors]).min()), axis=1)

    words[words["cluster_value"] == -1].sort_values("closeness_score")
    words[words["cluster_value"] == 0].sort_values("closeness_score")
    words[words["cluster_value"] == 1].sort_values("closeness_score")

    positive = ['zapewnia', 'zdrowi', 'zapewni', 'uchroni', 'dobri', 'znakomici', 'ciekawi', 'potwierdzwni', 'aktywni',
                'pierwszi']
    neutral = ['ogosio', 'zalecenia', 'pokazuj', 'przypominami', 'wrócić', 'przygotowao']
    negative = ['podejrzeniem', 'śmiertelnych', 'fake', 'brakować', 'pomylio', 'tragedia', 'lekceważci', 'opór', 'boję',
                'pomylio']

    for i in positive:
        words.loc[words["words"] == i, "cluster_value"] = 1

    for i in neutral:
        words.loc[words["words"] == i, "cluster_value"] = 0

    for i in negative:
        words.loc[words["words"] == i, "cluster_value"] = -1

    emotion = {0: "neutral",
               1: "positive",
               -1: "negative"}

    words["sentiments"] = words["cluster_value"].map(emotion)
    words_dict = dict(zip(words.words, words.cluster_value))

    dataset["sentiment"] = dataset[Constants.COL_TWEET_TEXT].apply(lambda text: get_sentiments(text, words_dict))

    print(f"{dataset['sentiment'].value_counts()}")

    for key, value in words_dict.items():
        print(f'{key} = {value}')

    return dataset


# Checks if code is run from google colan.
# Returns true if it is, and false if it isn't.
def is_run_from_co_lab():
    return 'google.colab' in str(get_ipython())
