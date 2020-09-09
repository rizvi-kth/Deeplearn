import numpy as np
import pandas as pd
import os
import itertools
import torch

# For IPython in PyCharm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
os.getcwd()
os.chdir("NLP/BERT/test_sw")
os.getcwd()

# Experiment 1
# ============
from transformers import pipeline

text = '''Römosseskolan, en skola med muslimsk inriktning i stadsdelen Gårdsten i Göteborg har bedrivit könsseparerad undervisning sedan skolan startades för 22 år sedan. Bland annat har pojkar och flickor haft separata lektioner i idrott, slöjd och musik, skriver SVT Väst som har tagit del av Skolinspektionens granskning av skolan.
Den svenska sommaren gör sig som bekant bäst i tanken. Minnesbild kan fogas till minnesbild tills idyll uppstår där ingen idyll i verkligheten fanns.
Det är så typiskt att Johannes Anyurus främsta sommardikt, ”Genom sjön, genom vattnet”, utspelar sig på hösten.'''

nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
out_list = nlp(text)


l = []
for token in out_list:
    if token['word'].startswith('##'):
        l[-1]['word'] += token['word'][2:]
    else:
        l += [ token ]

l


# new_tokens, new_labels = [], []
# for ent in out_list:
#
#     token = ent['word']
#     label = ent['entity']
#
#     if token.startswith("##"):
#         print(ent['word'])
#         new_tokens[-1] = new_tokens[-1] + token[2:]
#     else:
#         new_labels.append(label)
#         new_tokens.append(token)
#
# for w, l in zip(new_tokens, new_labels):
#     print(w, " -- ", l)


        # new_tokens, new_labels = [], []
        # for token, label_idx in zip(input_tokens, predicted_label_ids[0]):
        #     if token.startswith("##"):
        #         new_tokens[-1] = new_tokens[-1] + token[2:]
        #     else:
        #         new_labels.append(self.unique_tag_values[label_idx])
        #         new_tokens.append(token)

