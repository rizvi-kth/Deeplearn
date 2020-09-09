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


# Experiment 2
# ============
id2label = {
    "0": "O",
    "1": "OBJ",
    "2": "TME",
    "3": "ORG/PRS",
    "4": "OBJ/ORG",
    "5": "PRS/WRK",
    "6": "WRK",
    "7": "LOC",
    "8": "ORG",
    "9": "PER",
    "10": "LOC/PRS",
    "11": "LOC/ORG",
    "12": "MSR",
    "13": "EVN"
  }

unique_tag_values = list(id2label.values())

# Initialize the BERT classifier
# ==============================
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import AutoModel, AutoTokenizer
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
model = BertForTokenClassification.from_pretrained("KB/bert-base-swedish-cased-ner", num_labels=14, output_attentions=False, output_hidden_states=False)
# model = AutoModel.from_pretrained('KB/bert-base-swedish-cased-ner')
# Todo: Tell pytorch to run this model on the GPU.
# model.cuda()
model

# Load the model class
# ====================
import torch
# Model class must be defined somewhere
PATH = "./models/pytorch_model.bin"
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))

for k in checkpoint.keys():
    print(k)

model.load_state_dict(checkpoint)
model.eval()



# BERT Tokenization and Label mapping
# ===================================
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
print(torch.__version__)


MAX_LEN = 512

tokenizer = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased-ner', do_lower_case=False)


# Test the model with single sentence
# ===================================
model.eval()

test_sentence = """
Römosseskolan, en skola med muslimsk inriktning i stadsdelen Gårdsten i Göteborg har bedrivit könsseparerad undervisning sedan skolan startades för 22 år sedan. Bland annat har pojkar och flickor haft separata lektioner i idrott, slöjd och musik, skriver SVT Väst som har tagit del av Skolinspektionens granskning av skolan. 
Utifrån de uppgifter vi har fått har man tillämpat nån typ av könsseparerad undervisning på den här skolan under lång tid ja, säger Frida Eek, jurist vid Skolinspektionen till SVT Väst.
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







