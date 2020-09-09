import json
import logging
import os
import numpy as np

import torch
# from transformers import BertForTokenClassification
# from transformers import BertTokenizer
from transformers import pipeline

# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from ts.torch_handler.base_handler import BaseHandler
# class NerClassifierHandler(BaseHandler):

logger = logging.getLogger(__name__)


class SvNerHandler(object):
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
        logger.info(f'{">>>" * 20}\n Server Device : {"cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"}')
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Get model-weight directory
        # model_dir = properties.get("model_dir")
        # logger.info(f'{">>>" * 20}\n Transformer model from path : {model_dir} ')

        # Get Bert model
        model = nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
        logger.info(f'{">>>" * 20}\n SV NER model loaded successfully')
        self.model = model

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentence = text.decode('utf-8')
        logger.info("\n>>> Received text: '%s'", sentence)
        return sentence

    def inference(self, input_sentence):
        # Input the IDs to model to predict label-IDs
        with torch.no_grad():
            out_list = self.model(input_sentence)

        l = []
        for token in out_list:
            if token['word'].startswith('##'):
                l[-1]['word'] += token['word'][2:]
            else:
                l += [token]

        return l

    def postprocess(self, inference_output):
        return [inference_output]


_service = SvNerHandler()


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
