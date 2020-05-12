import numpy as np
import sys
import os
import itertools
import pandas as pd
from transformers import BertTokenizer
import torch

# For IPython in PyCharm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
os.getcwd()
os.chdir("NLP/BERT/test1")
os.getcwd()

# To make available the local modules and data files with relative paths
import sys
sys.path.append("./")

# Reading CONLL-2003 data for NER from drive.
# ==========================================
path_train_cola = "./data/CONLL_ENG_NER_2003/ner_only/train_cola.csv"
# Load the dataset into a pandas dataframe.
df = pd.read_csv(path_train_cola, encoding='utf-8')
# Report the number of sentences.
print('Number of training words: {:,}\n'.format(df.shape[0]))
# Display 10 random rows from the data.
df.head(5)


# Check the null values and clean them.
# ====================================
# Show the number of rows with any of the value is None
print(f'Number of rows with any NaN value :', df.isna().T.any().sum())
# Show the rows with any of the value is None
df[df.isna().T.any().T]
# Drop row with any of the value being NaN.
df.dropna(how='any', inplace=True)
print('Number of training words after null drop: {:,}\n'.format(df.shape[0]))


# Check out the unique Name-Entity and generate dictionary
# ========================================================
unique_entity_us = df['NE'].unique()
unique_entity_us.sort()
unique_entity = list(unique_entity_us)
# unique_entity.append('[SEP]')
unique_entity.append('<CLS>')
print("Name entity Labels : ", unique_entity)
# Set up IDs for labels
name_entity_id = [50, 60, 40, 30, 55, 66, 44, 33, 0, 1]
assert len(name_entity_id) == len(unique_entity), "Name entity and corresponding IDs should of same number"
# Construct dictionary with name entities and IDs
dict_name_entity = dict(zip(unique_entity, name_entity_id))
print("Name entity Label-ids : ", dict_name_entity)

# Reform the Dataframe to List-of-words
# ========================================
# Convert pandas column to arrays
word_list = df.WORDS.values
print("\nWords list count: ", len(word_list))
print(word_list[0:9])
# Convert pandas column to arrays
entity_list = df.NE.values
print("\nEntity list count: ", len(entity_list))
print(entity_list[0:9])
assert len(word_list) == len(entity_list), "Word list and the entity list should be equal."


# Make sentence-list out of word-list
# ===================================
# Split the word list on '.' and make a sentence-list. Make the same kind of split on the name entity-list.
ents_list = []
ents_4_sents = []
sent = []
sentences = []
for w, e in zip(word_list, entity_list):
    sent.append(w)
    ents_list.append(e)
    if w == ".":
        sentences.append(sent)
        ents_4_sents.append(ents_list)
        sent = []
        ents_list = []

assert len(sentences) == len(ents_4_sents), "Sentence list and corrosponding entity list should be equal."
print("Total number of sentances (word-list): ", len(sentences))
print("Total number of entity-list          : ", len(ents_4_sents))
print("\nSample sentances: ")
print(sentences[:1])
print("\nSample entity-list: ")
print(ents_4_sents[:1])
# Show some sample sentences with their name-entity tags.
for s, es in zip(sentences[:10], ents_4_sents[:10]):
  print("\nSentence : ")
  for w, ew in zip(s, es):
    print(f"{w}({ew}) ", end=" ")


# Load the BERT tokenizer.
# ========================
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


# Function for rearranging name-entity
# ==================================


def post_encoding_rearrange_name_entity_list(encoded_input_ids, sentence_arr, ne_list):
    print("In function Sentence : ", sentence_arr)
    # Decode back to sentence-strings
    decoded_string = tokenizer.decode(token_ids=encoded_input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded_arr = decoded_string.split(' ')
    print(' In function Decoded : ', decoded_arr)
    # Rearrange the name-entity list(the labels) so that it maps to the sentence-ids(the input).
    assert len(decoded_arr) >= len(sentence_arr), "Decoded sentence should be >= original sentence."

    rearranged_ne = []
    sentence_arr.insert(0, "[CLS]")
    ne_list.insert(0, "<CLS>")
    for idx, (dec_w, org_w) in enumerate(itertools.zip_longest(decoded_arr, sentence_arr, fillvalue=None)):
        if dec_w == org_w:
            rearranged_ne.insert(idx, ne_list[idx])
        else:
            rearranged_ne.insert(idx, "O")
        # print(f"{idx}\t{dec_w}\t{org_w}\t{rearranged_ne[idx]}\t")
    print("In function Align NE : ", rearranged_ne)
    assert len(rearranged_ne) == len(encoded_input_ids), "Inputs should be same length of labels"
    return rearranged_ne


# Whole Sentence-list Tokenization
# ================================
input_ids = []
attention_masks = []
labels_ne = []
labels_ne_id = []
# For every sentence...
for indx, (sentence_arr, ne_list) in enumerate(zip(sentences, ents_4_sents)):
    print(indx)
    print("Input Sentance     : ", sentence_arr)

    # Tokenize the sentence and map the tokens to their word-IDs.
    encoded_dict = tokenizer.encode_plus(
        sentence_arr,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # print(' Encoded : ', encoded_dict['input_ids'])
    # raise KeyboardInterrupt
    print("Sentance Lenght      : ", len(sentence_arr))
    print("Encoded Tensor shape : ", encoded_dict['input_ids'].shape)

    # Rearrange the name-entity list(the labels) so that it maps to the sentence-ids(the input).
    rearranged_ne = post_encoding_rearrange_name_entity_list(encoded_dict['input_ids'].squeeze(),
                                                             sentence_arr[:encoded_dict['input_ids'].shape[1] - 1].copy(),
                                                             ne_list[:encoded_dict['input_ids'].shape[1] - 1].copy())
    labels_ne.append(rearranged_ne)
    # Assign IDs to lebels
    rearranged_ne_id = [dict_name_entity[en] for en in rearranged_ne]
    labels_ne_id.append(rearranged_ne_id)
    print("Out function Alig.NE : ", rearranged_ne_id)

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])


# Transform the labels to Label-IDs
# =================================
# labels_ne_id = []
# for en_arr in labels_ne:
#     labels_ne_id.append([dict_name_entity[en] for en in en_arr])


# Convert the lists into tensors.
# ===============================
input_ids_tn = torch.cat(input_ids, dim=0)
attention_masks_tn = torch.cat(attention_masks, dim=0)
labels_ne_ids_tn = torch.tensor(labels_ne_id)
print("    Input shape : ", input_ids_tn.shape)
print("Att. mask shape : ", attention_masks_tn.shape)
print("   Output shape : ", labels_ne_ids_tn.shape)

# raise KeyboardInterrupt
# [sen[:3] for sen in  sentences[50:59]]
