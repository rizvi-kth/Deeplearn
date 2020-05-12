import pandas as pd


class DataGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["word"].values.tolist(),
                                                           s['pos'].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]





def get_sentences_labels_tags(path):
    df = pd.read_csv(path, encoding="ISO-8859-1", error_bad_lines=False)
    df.rename(columns={df.columns[0]: "sentence_idx",
                       df.columns[1]: "word",
                       df.columns[2]: "pos",
                       df.columns[3]: "tag"}, inplace=True)
    df['sentence_idx'].ffill(axis=0, inplace=True)
    df.tail(50)

    getter = DataGetter(df)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    print('First sentence: ', sentences[0])

    labels = [[s[2] for s in sentence] for sentence in getter.sentences]
    print('First label: ', labels[0])

    unique_tag_values = list(set(df["tag"].values))
    print('Unique tag values: ', unique_tag_values)

    return sentences, labels, unique_tag_values
