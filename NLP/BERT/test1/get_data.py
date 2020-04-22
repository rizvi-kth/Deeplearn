import wget
import os

print('Downloading dataset...')

# The URL for the dataset zip file.
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
url_2 = 'http://www.cnts.ua.ac.be/conll2003/eng.raw.tar'

# Download the file (if we haven't already)
if not os.path.exists('./data/cola_public_1.1.zip'):
    wget.download(url, './data/cola_public_1.1.zip')

if not os.path.exists('./data/eng.raw.tar'):
    wget.download(url, './data/eng.raw.tar')


# Unzio in cli
# unzip cola_public_1.1.zip

# Read data and check the samples

import pandas as pd


# Load the dataset into a pandas dataframe.
df = pd.read_csv("./data/cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
print(df.sample(10))