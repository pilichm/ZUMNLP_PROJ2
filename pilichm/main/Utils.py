import re

import numpy as np
import pandas as pd
import spacy
from IPython import get_ipython
from matplotlib import pyplot as plt
from nltk import PorterStemmer
import multiprocessing
import pl_core_news_sm
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.cluster import KMeans

from pilichm.main.Constants import Constants


# Returns sets of words associated with positive, negative and neutral tweets.
# Number of sentences for each sentiment should be the same.
def get_pos_neg_neut_words():
    positive_sentences = [
        'bardzo ciekawi wywiad',
        'ciekawa teoria coronaviru stworzoni czowieka',
        'bardzo dobra decyzja bo wiru laboratorium wuhan',
        'dobr wiadomości pierwszej linii frontu',
        'dobr wieści najwyraźniej przeciwko coronaviruscovid wedug ku leuven lek przeciw malarii wprowadzoni',
        'koronawiru polaci wrócili wuhan opuszczają szpital',
        'czyli polaci mądrzejsi'
    ]

    neutral_sentences = [
        'co wiemi pori',
        'kolejn czasopismo medyczn przygotowao',
        'dokadni opisuj niektóri',
        'pod adresem email lekarz udzi ogólnych informacji temat',
        'obrazki targu wuhan',
        'potrzebujesz informacji pomoci koronawiru covid',
        'we woszech sytuacja rozwojowa coronaviru'
    ]

    negative_sentences = [
        'boję wirusa',
        'lekce ważci tragedia coronaviru',
        'tak obecni wygląda wuhan przerażająci',
        'uważaj kretyni żebyś apiąc covid zapiejesz ćwoku inaczej',
        'zmari leżąci  ludzi bali podejść',
        'pani za drogo nikt wuhan kupi',
        'jeszcz chwila stracą kontrolę sytuacją'
    ]

    positive, negative, neutral = set(), set(), set()

    for pos_sent, neg_sent, neut_sent in zip(positive_sentences, negative_sentences, neutral_sentences):
        for word in pos_sent:
            positive.add(word)

        for word in neg_sent:
            negative.add(word)

        for word in neut_sent:
            neutral.add(word)

    return positive, negative, neutral


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
def clean_data(path_to_csv):
    url_pattern = re.compile(Constants.REGEX_URL)
    letters_pattern = re.compile(Constants.REGEX_POLISH_CHARS)
    nlp = pl_core_news_sm.load()
    df = pd.read_csv(path_to_csv, delimiter='\t')

    # Remove urls.
    df[Constants.COL_CLEANED_TEXT] = df[Constants.COL_TWEET_TEXT] \
        .apply(lambda text: url_pattern.sub(r' ', text))

    # Remove emails.
    df[Constants.COL_CLEANED_TEXT] = df[Constants.COL_CLEANED_TEXT] \
        .apply(lambda text: re.sub(Constants.REGEX_EMAIL, ' ', text))

    # Remove all characters that aren't polish letters.
    df[Constants.COL_CLEANED_TEXT] = df[Constants.COL_CLEANED_TEXT] \
        .apply(lambda text: letters_pattern.sub(r'', text))

    # Remove stop words.
    df[Constants.COL_CLEANED_TEXT] = df[Constants.COL_CLEANED_TEXT] \
        .apply(lambda text: remove_stop_and_single_words_and_stem(text, nlp))

    # Remove repetitions.
    df[Constants.COL_CLEANED_TEXT] = df[Constants.COL_CLEANED_TEXT] \
        .apply(lambda text: remove_repetitions(text))

    # Tokenize - stemming.
    df[Constants.COL_CLEANED_TEXT] = df[Constants.COL_CLEANED_TEXT] \
        .apply(lambda text: stem_line(text).strip())

    df.dropna(inplace=True)
    df.to_csv(path_to_csv, sep='\t', index=False)

    df = pd.read_csv(path_to_csv, delimiter='\t')
    df.drop(df[pd.isnull(df[Constants.COL_CLEANED_TEXT])].index, inplace=True)
    df.to_csv(path_to_csv, sep='\t', index=False)


def get_sentiments(text, words_dict):
    # print(f'Current line >{text}<')

    total = 0
    count = 0
    print(f'Current: >{text}<')
    for t in text.split(' '):
        # print(f'Current word >{t}<')
        if words_dict.get(t):
            # print(f'MATCH {words_dict.get(t)} for {t}')
            total += int(words_dict.get(t))
        count += 1

    # print(f'total {total} count {count}')
    if count > 0:
        avg = total / count
    else:
        avg = 0

    # print(f'{avg} = {total} / {count}')

    sentiment = -1 if avg < -0.01 else 1 if avg > 0.01 else 0
    return sentiment


# Add sentiment labels to tweets in dataset.
def create_labels(path_to_csv, num_of_classes=3):
    # Create word embeddings.
    df = pd.read_csv(path_to_csv, delimiter='\t')
    sent = [row for row in df[Constants.COL_CLEANED_TEXT] if not pd.isna(row)]
    # for line in sent:
    #     print(f'Line {line} | of type {type(line)}')
    phrases = Phrases(sent, min_count=1, progress_per=50000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    print(type(sentences[1]))
    print(sentences[1])

    for idx, sentence in enumerate(sentences):
        sentences[idx] = sentence.split()

    print(type(sentences[1]))
    print(sentences[1])

    w2v_model = Word2Vec(min_count=4, window=5, sample=1e-5, alpha=0.03, min_alpha=0.0007,
                         negative=20, seed=42, workers=multiprocessing.cpu_count() - 1)

    w2v_model.build_vocab(sentences, progress_per=50000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save("word2vec.model")

    word_vectors = Word2Vec.load("word2vec.model").wv
    model = KMeans(n_clusters=num_of_classes, max_iter=1000, random_state=42, n_init=50).fit(
        X=word_vectors.vectors.astype('double'))
    word_vectors.similar_by_vector(model.cluster_centers_[1], topn=200, restrict_vocab=None)

    # print('MOST SIMILAR FOR 0')
    # print(word_vectors.similar_by_vector(model.cluster_centers_[0], topn=25, restrict_vocab=None))
    #
    # for word, prob in word_vectors.similar_by_vector(model.cluster_centers_[0], topn=25, restrict_vocab=None):
    #     print(f'WORD >{word}<')
    #
    # print('MOST SIMILAR FOR 1')
    # print(word_vectors.similar_by_vector(model.cluster_centers_[1], topn=25, restrict_vocab=None))
    #
    # print('MOST SIMILAR FOR 2')
    # print(word_vectors.similar_by_vector(model.cluster_centers_[2], topn=25, restrict_vocab=None))

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

    positive, negative, neutral = get_pos_neg_neut_words()
    # positive, negative, neutral = [], [], []
    #
    # for word, prob in word_vectors.similar_by_vector(model.cluster_centers_[0], topn=100, restrict_vocab=None):
    #     positive.append(word)
    #
    # for word, prob in word_vectors.similar_by_vector(model.cluster_centers_[1], topn=100, restrict_vocab=None):
    #     negative.append(word)
    #
    # for word, prob in word_vectors.similar_by_vector(model.cluster_centers_[2], topn=100, restrict_vocab=None):
    #     neutral.append(word)

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

    df["sentiment"] = df[Constants.COL_CLEANED_TEXT].apply(lambda text: get_sentiments(text, words_dict))

    print(f"{df['sentiment'].value_counts()}")

    # for key, value in words_dict.items():
    #     print(f'{key} = {value}')

    df.to_csv(path_to_csv, sep='\t', index=False)


# Checks if code is run from google colan.
# Returns true if it is, and false if it isn't.
def is_run_from_co_lab():
    return 'google.colab' in str(get_ipython())


def display_sentiment_distribution(path_to_csv):
    emotion = {0: "neutral",
               1: "positive",
               -1: "negative"}

    df = pd.read_csv(path_to_csv, delimiter='\t')
    df["sentiment"] = df["sentiment"].map(emotion)
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    colors = ["cyan", "pink", "yellow"]
    df_pie = df["sentiment"].value_counts().reset_index()
    plt.pie(df_pie["sentiment"], labels=df_pie["index"], radius=2, colors=colors, autopct="%1.1f%%")
    plt.axis('equal')
    plt.title("Dystrybucja sentymentów ", fontsize=20)
    plt.show()
