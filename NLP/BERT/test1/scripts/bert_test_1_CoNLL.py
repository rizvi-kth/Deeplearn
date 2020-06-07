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


# Read Data file
# ==============
PAD_LABEL = "PAD"
import SentenceLoaderCoNLL as sg
sentences_train, labels_train, unique_tag_values_train = sg.get_sentences_labels_tags("./data/CONLL_ENG_NER_2003/ner_only/train_cola.csv")
sentences_test, labels_test, unique_tag_values_test = sg.get_sentences_labels_tags("./data/CONLL_ENG_NER_2003/ner_only/test_cola.csv")
print(f"Training samples : {len(sentences_train)}  \nTesting samples : {len(sentences_test)} ")
TRAIN_SIZE = len(sentences_train)

# Concatenate train and test to a single list
sentences_train.extend(sentences_test)
sentences = sentences_train
labels_train.extend(labels_train)
labels = labels_train

assert set(unique_tag_values_train) == set(unique_tag_values_test), "The Train dataset and Test dataset dont have the same labels."
unique_tag_values = list(set(unique_tag_values_train)) # or unique_tag_values_test can be taken

unique_tag_values.append(PAD_LABEL)
tag2idx = {t: i for i, t in enumerate(unique_tag_values)}
print(f"The TAG dictionary : {tag2idx}")


# BERT Tokenization and Label mapping
# ===================================
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
print(torch.__version__)


MAX_LEN = 75
bs = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
print(tokenized_texts_and_labels[0][0][5:9])
print(tokenized_texts_and_labels[0][1][5:9])
tokenized_sentences = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels_4_sentence = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


# BERT encoding
# =============
input_ids = []
attention_masks = []
labels_nes = []
labels_ne_ids = []
# For every sentence...
# Todo : Temporarily taking 100 sentences. Take all sentences.
for indx, (sentence_arr, ne_list) in enumerate(zip(tokenized_sentences[:100], labels_4_sentence[:100])):
    print( "=== ",indx, " =========================================================================" )
    print("Input Sentance     : ", sentence_arr)

    # Tokenize the sentence and map the tokens to their word-IDs.
    encoded_dict = tokenizer.encode_plus(
        sentence_arr,  # Sentence to encode.
        add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # print(' Encoded : ', encoded_dict['input_ids'])
    # raise KeyboardInterrupt
    print("     Sentance Lenght : ", len(sentence_arr))

    # !!! Decoded sentence will not be aligned with label
    # decoded_sent = tokenizer.decode(token_ids=encoded_dict["input_ids"].squeeze(),
    #                                                    skip_special_tokens=False,
    #                                                    clean_up_tokenization_spaces=False)
    # print("Decoded Sente. Shape :  ", len(decoded_sent.split()))
    # print("    Decoded Sentence :  ", decoded_sent)
    # print("     Sentence Label  :  ", ne_list)
    sentence_arr = sentence_arr[0:MAX_LEN]
    ne_list = ne_list[0:MAX_LEN]
    print("(word, e-id, lbl)    :  ", [(wd, ev.item(), ne) for wd, ev, ne in itertools.zip_longest(sentence_arr, encoded_dict["input_ids"].squeeze(), ne_list)])
    ne_list_padded = [ne if ev.item() != 0 else PAD_LABEL for ev, ne in itertools.zip_longest(encoded_dict["input_ids"].squeeze(), ne_list)]
    print("(word, e-id, lbl-pad):  ", [(wd, ev.item(), ne) for wd, ev, ne in itertools.zip_longest(sentence_arr, encoded_dict["input_ids"].squeeze(), ne_list_padded)])
    labels_ne_id = [tag2idx[le] for le in ne_list_padded]
    print("(word, e-id, lbl-id) :  ", [(wd, ev.item(), ne) for wd, ev, ne in itertools.zip_longest(sentence_arr, encoded_dict["input_ids"].squeeze(), labels_ne_id)])

    # Add Input-Ids
    print("Encoded Tensor shape : ", encoded_dict['input_ids'].shape)
    print("           Input IDs :  ", encoded_dict["input_ids"])
    input_ids.append(encoded_dict["input_ids"])

    # Add Attention mask
    print("Encoded Atten. shape : ", encoded_dict['attention_mask'].shape)
    print("      Attention Mask :  ", encoded_dict["attention_mask"])
    attention_masks.append(encoded_dict["attention_mask"])

    # Add Label-IDs
    labels_ne_id_t = torch.tensor(labels_ne_id).unsqueeze(dim=0)
    print("Encoded Lbl-id shape : ", labels_ne_id_t.shape)
    print("            Label-id :  ", labels_ne_id_t)
    labels_ne_ids.append(labels_ne_id_t)


# Convert the lists into tensors.
# ===============================
input_ids_tn = torch.cat(input_ids, dim=0)
attention_masks_tn = torch.cat(attention_masks, dim=0)
labels_ne_ids_tn = torch.cat(labels_ne_ids, dim=0)
print("    Input shape : ", input_ids_tn.shape)
print("Att. mask shape : ", attention_masks_tn.shape)
print("   Output shape : ", labels_ne_ids_tn.shape)


# Split train and validation set
# ==============================
# from torch.utils.data import TensorDataset, random_split
# # Combine the training inputs into a TensorDataset.
# dataset = TensorDataset(input_ids_tn, attention_masks_tn, labels_ne_ids_tn)
# # Create a 95-10 train-validation split.
# # Calculate the number of samples to include in each set.
# train_size = int(0.95 * len(dataset))
# val_size = len(dataset) - train_size
# # Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))

# Todo: Remove when considering the whole dataset
TRAIN_SIZE = 75

train_dataset = TensorDataset(input_ids_tn[0:TRAIN_SIZE-1, :], attention_masks_tn[0:TRAIN_SIZE-1, :], labels_ne_ids_tn[0:TRAIN_SIZE-1, :])
val_dataset   = TensorDataset(input_ids_tn[TRAIN_SIZE-1: , :], attention_masks_tn[TRAIN_SIZE-1: , :], labels_ne_ids_tn[TRAIN_SIZE-1: , :])

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))


# Encapsulate in data loader
# ==========================
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
BATCH_SIZE = 32
# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=BATCH_SIZE  # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=BATCH_SIZE  # Evaluate with this batch size.
        )


# Initialize the BERT classifier
# ==============================
from transformers import BertForTokenClassification, AdamW, BertConfig
# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(unique_tag_values), output_attentions=False, output_hidden_states=False)
# Todo: Tell pytorch to run this model on the GPU.
# model.cuda()
model


# Get all of the model's parameters
# =================================
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# Optimizer and lr scheduler
# ==========================
# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
from transformers import get_linear_schedule_with_warmup
# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the training data.
EPOCHS = 4
# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * EPOCHS
print("Total number of training steps", total_steps)
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Test single Training step
# =========================
model.train()
total_train_loss = 0

train_input = next(iter(train_dataloader))[0]
train_atten = next(iter(train_dataloader))[1]
train_output = next(iter(train_dataloader))[2]
# Todo : Tell pytorch to run this model in GPU
# train_input = train_input.to(device)
# train_atten = train_atten.to(device)
# train_output = train_output.to(device)
print("Train input shape : ", train_input.shape)
print("Train attention shape : ", train_atten.shape)
print("Train label shape : ", train_output.shape)


model.zero_grad()
loss, logits = model(train_input, token_type_ids=None, attention_mask=train_atten, labels=train_output)
total_train_loss += loss.item()
# Perform a backward pass to calculate the gradients.
loss.backward()
# Clip the norm of the gradients to 1.0.
# This is to help prevent the "exploding gradients" problem.
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# Update parameters and take a step using the computed gradient.
# The optimizer dictates the "update rule"--how the parameters are
# modified based on their gradients, the learning rate, etc.
optimizer.step()
# Update the learning rate.
scheduler.step()


# Test single Validation step
# ===========================
model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
predictions, true_labels = [], []

batch = next(iter(validation_dataloader))
b_input_ids, b_input_mask, b_labels = batch
with torch.no_grad():
    # Forward pass, calculate logit predictions.
    # This will return the logits rather than the loss because we have not provided labels.
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  # labels=b_labels

# Move logits and labels to CPU
logits = outputs[0].numpy()
label_ids = b_labels.numpy()

# eval_accuracy += flat_accuracy(logits, label_ids)
# pred_flat = np.argmax(logits, axis=2).flatten()
# labels_flat = labels.flatten()
# return np.sum(pred_flat == labels_flat) / len(labels_flat)

predictions = [list(p) for p in np.argmax(logits, axis=2)]
true_labels = label_ids

pred_tags = [unique_tag_values[p_i] for p, l in zip(predictions, true_labels)
                                    for p_i, l_i in zip(p, l) if unique_tag_values[l_i] != "PAD"]
valid_tags = [unique_tag_values[l_i] for l in true_labels
                                  for l_i in l if unique_tag_values[l_i] != "PAD"]

from seqeval.metrics import f1_score
print("Validation F1-Score (without PAD) : {}".format(f1_score(pred_tags, valid_tags)))













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
    print("{}\t{}".format(label, token))



# Saving the model
# ================
PATH = "./models/bert_CoNLL_model.pth"
torch.save(model.state_dict(), PATH)

