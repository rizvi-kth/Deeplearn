import numpy as np
import pandas as pd
import os
import itertools

# For IPython in PyCharm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
os.getcwd()
os.chdir("TorchServe/BERT_deploy")
os.getcwd()

# To make available the local modules and data files with relative paths
# import sys
# sys.path.append("./")
# sys.path.append("./scripts")

# Load the model class
# ====================
import torch
# Model class must be defined somewhere
PATH = "./../../NLP/BERT/test1/models/bert_CoNLL_model_colab_2.pth"
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
unique_tag_values = checkpoint['model_class']
print("Model classes : ", unique_tag_values)




