import json
import logging
import os
import numpy as np

import torch
from transformers import BertForTokenClassification
from transformers import BertTokenizer

# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from ts.torch_handler.base_handler import BaseHandler
# class NerClassifierHandler(BaseHandler):

NUM_LABELS = 10
MAX_LEN = 75
logger = logging.getLogger(__name__)


class NerClassifierHandler(object):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.unique_tag_values = None
        self.batch_size = None

    def initialize(self, ctx):
        # Get properties from context
        properties = ctx.system_properties
        self.batch_size = properties["batch_size"]
        logger.info(f'{">>>" * 20}\n Batch-size     : {self.batch_size}')
        logger.info(f'{">>>" * 20}\n Server name    : {properties["server_name"]}')
        logger.info(f'{">>>" * 20}\n Server version : {properties["server_version"]}')
        logger.info(f'{">>>" * 20}\n GPU-id         : {properties["gpu_id"]}')

        # Prepare the DEVICE from context
        logger.info(f'{">>>" * 20 }\n Server Device : {"cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"}')
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Get model-weight directory
        model_dir = properties.get("model_dir")
        logger.info(f'{">>>" * 20 }\n Transformer model from path : {model_dir} ')

        # Get Bert model
        model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=NUM_LABELS,
                                                           output_attentions=False, output_hidden_states=False)
        logger.info(f'{">>>" * 20 }\n BertForTokenClassification loaded successfully')

        # Load the weights from model-weight directory
        checkpoint = torch.load(os.path.join(model_dir, "bert_CoNLL_model_colab_2.pth"), map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f'{">>>" * 20 }\n NER model loaded successfully')
        self.model = model
        self.unique_tag_values = checkpoint['model_class']
        assert len(self.unique_tag_values) == NUM_LABELS, \
            "Model labels count should be equal to BertForTokenClassification models num_labels"

        # Get the Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        logger.info(f'{">>>" * 20 }\n BertTokenizer loaded successfully')
        self.tokenizer = tokenizer

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentence = text.decode('utf-8')
        logger.info("\n>>> Received text: '%s'", sentence)

        # Tokenize the sentence
        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LEN,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids = tokenized_sentence['input_ids']
        return input_ids

    def inference(self, input_ids):

        # Decode IDs back to tokens to see the original input.
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

        # Input the IDs to model to predict label-IDs
        with torch.no_grad():
            output = self.model(input_ids)
        predicted_label_ids = np.argmax(output[0].to('cpu').numpy(), axis=2)

        new_tokens, new_labels = [], []
        for token, label_idx in zip(input_tokens, predicted_label_ids[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.unique_tag_values[label_idx])
                new_tokens.append(token)

        for token, label in zip(new_tokens, new_labels):
            logger.info("{:5} \t {}".format(label, token))

        return [(new_tokens, new_labels)]

    def postprocess(self, inference_output):
        return inference_output


_service = NerClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        logger.error(f'{">>>" * 20}\n Error thrown from NerClassifierHandler')
        raise e


