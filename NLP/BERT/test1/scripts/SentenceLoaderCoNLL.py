import numpy as np
import pandas as pd
import sys

import os
import itertools

class DataGetterCoNLL(object):

    def __init__(self, dataframe):

        self.df = dataframe
        self.sentence_tag_list = [(w, t) for w, t in zip(self.df["WORDS"].values.tolist(), self.df['NE'].values.tolist())]
        self.sentence_list = []
        sentence = []
        # [(sentence.append(pair[0]), sentence_list.append(sentence.copy()), sentence.clear())  if pair[0] == "." else sentence.append(pair[0]) for pair in sent_tag_list]
        list(map(lambda pair: (sentence.append(pair), self.sentence_list.append(sentence.copy()), sentence.clear()) if pair[0] == "." else sentence.append(pair), self.sentence_tag_list))
        self.n_sentences = len(self.sentence_list)


def get_sentences_labels_tags(path):
    df = pd.read_csv(path, encoding="utf-8", error_bad_lines=True)
    print('Number of training words: {:,}'.format(df.shape[0]))
    # NULL drop
    df.dropna(how='any', inplace=True)
    print('Number of training words after null drop: {:,}\n'.format(df.shape[0]))

    # Unique entity list
    unique_entity_us = df['NE'].unique()
    unique_entity_us.sort()
    unique_entity = list(unique_entity_us)
    print("Name Entity : ", unique_entity)

    getter = DataGetterCoNLL(df)
    print("Number of sentences : ", getter.n_sentences)

    sentences = [[word[0] for word in sentence] for sentence in getter.sentence_list]
    print('First sentence: ', sentences[0])

    labels = [[s[1] for s in sentence] for sentence in getter.sentence_list]
    print('First label: ', labels[0])

    return sentences, labels, unique_entity


if __name__ == "__main__":
    # Run this script
    sys.path.append("./")
    get_sentences_labels_tags("./../data/CONLL_ENG_NER_2003/ner_only/train_cola.csv")
