import numpy as np
import pandas as pd
import os
import itertools

# For IPython in PyCharm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
os.getcwd()
os.chdir("NLP/BERT/test1")
os.getcwd()

# To make available the local modules and data files with relative paths
import sys
sys.path.append("./")
sys.path.append("./scripts")

# Get the Classifier
# ==================
# unique_tag_values = ['I-PER',
#  'B-MISC',
#  'I-ORG',
#  'O',
#  'B-ORG',
#  'I-MISC',
#  'B-PER',
#  'B-LOC',
#  'I-LOC',
#  'PAD']

# Load the model class
# ====================
import torch
# Model class must be defined somewhere
PATH = "./models/bert_CoNLL_model_colab_2.pth"
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
unique_tag_values = checkpoint['model_class']
unique_tag_values


# Initialize the BERT classifier
# ==============================
from transformers import BertForTokenClassification, AdamW, BertConfig
# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(unique_tag_values), output_attentions=False, output_hidden_states=False)
# Todo: Tell pytorch to run this model on the GPU.
# model.cuda()
model


# Load the model
# ==============
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# BERT Tokenization and Label mapping
# ===================================
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
print(torch.__version__)


MAX_LEN = 75
bs = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


# Test the model with single sentence
# ===================================
model.eval()

test_sentence = """
Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
reporter for the network, about protests in Minnesota and elsewhere. 
"""

tokenized_sentence = tokenizer.encode_plus(
        test_sentence,  # Sentence to encode.
        add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=False,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
# Todo: move the ids to GPU
# input_ids = torch.tensor([tokenized_sentence]).cuda()
input_ids = tokenized_sentence['input_ids']

# Decode IDs back to tokens to see the original input.
input_tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

# Input the IDs to model to predict label-IDs
with torch.no_grad():
    output = model(input_ids)
predicted_label_ids = np.argmax(output[0].to('cpu').numpy(), axis=2)


new_tokens, new_labels = [], []
for token, label_idx in zip(input_tokens, predicted_label_ids[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(unique_tag_values[label_idx])
        new_tokens.append(token)

for token, label in zip(new_tokens, new_labels):
    print("{:5} \t {}".format(label, token))
