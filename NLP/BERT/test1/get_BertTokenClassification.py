from transformers import BertForTokenClassification, AdamW, BertConfig

# Load BertForTokenClassification, the pretrained BERT model with a single
# ========================================================================
# linear classification layer on top.
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=64,  # Number of output labels-2 for binary classification. Increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
print(model)

# Examine teh Model
# =================
# Get all of the model's parameters as a list of tuples.
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