import torch
from torch import nn, optim
import torchtext
import spacy

if __name__ == '__main__':
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_web_sm')