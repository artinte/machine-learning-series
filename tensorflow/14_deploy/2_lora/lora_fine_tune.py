import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_hub
import keras
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds

# General hyperparameters
BATCH_SIZE = 32
NUM_BATCHES = 500
EPOCHES = 1 # can be set to a higher value for better results
MAX_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 200

GPT2_PRESET = 'gpt2_base_en'

# LoRA-specific hyperparameters
RANK = 4
ALPHA = 32.0

reddit_ds = tfds.load('reddit_tifu', split='train', as_supervised=True)

for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break
