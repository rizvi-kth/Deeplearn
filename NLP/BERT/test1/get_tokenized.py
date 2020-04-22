from transformers import BertTokenizer
import itertools

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')

# Input for BERT model with Capital-Case letters
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
sentence_arr = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']


# # Input for BERT model with Small-Case letters
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# sentence_arr_cased = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
# sentence_arr = [w.lower() for w in sentence_arr_cased]


# The labels
ne_list = ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
print('   Labels: ', ne_list)
print(' Original: ', sentence_arr)


# token_list = tokenizer.convert_tokens_to_ids(sentence_arr)
# print('Token IDs: ', token_list)
# assert len(token_list) == len(sentence_arr), "Token token_list and sentence_arr miss match."
# assert len(token_list) == len(ne_list), "Token token_list and ne_list miss match."


# Tokenize the sentence and map the tokens to their word-IDs.
input_ids = []
attention_masks = []
encoded_dict = tokenizer.encode_plus(
                                    sentence_arr,  # Sentence to encode.
                                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                    max_length=64,  # Pad & truncate all sentences.
                                    pad_to_max_length=True,
                                    return_attention_mask=True,  # Construct attn. masks.
                                    return_tensors='pt',  # Return pytorch tensors.
                                )
print(' Encoded : ', encoded_dict['input_ids'][0])


# Decode back to sentence-strings
decoded_string = tokenizer.decode(token_ids=encoded_dict['input_ids'][0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
decoded_arr = decoded_string.split(' ')
print(' Decoded : ', decoded_arr)


# Rearrange the name-entity list(the labels) so that it maps to the sentence-ids(the input).
rearranged_ne = []
sentence_arr.insert(0, None)
ne_list.insert(0, "O")
assert len(decoded_arr) > len(sentence_arr), "Decoded sentence can't be longer than original sentence."
for idx, (dec_w, org_w) in enumerate(itertools.zip_longest(decoded_arr, sentence_arr, fillvalue=None)):

    if dec_w == org_w:
        rearranged_ne.insert(idx, ne_list[idx])
    else:
        rearranged_ne.insert(idx, "O")

    print(f"{idx}\t{dec_w}\t{org_w}\t{rearranged_ne[idx]}\t")
assert len(rearranged_ne) == len(encoded_dict['input_ids'][0]), "Inputs should be same length of labels"


# # Add the encoded sentence to the list.
# input_ids.append(encoded_dict['input_ids'])
# print(input_ids)
#
# # And its attention mask (simply differentiates padding from non-padding).
# attention_masks.append(encoded_dict['attention_mask'])
#
# # Convert the lists into tensors.
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# labels = torch.tensor(labels)
#
# # Print sentence 0, now as a list of IDs.
# print('Original: ', sentences[0])
# print('Token IDs:', input_ids[0])