import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt

# 添加 Preprocessor 到安全全局列表
from dp.preprocessing.text import Preprocessor
torch.serialization.add_safe_globals([Preprocessor])


torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.__version__)
print(torchaudio.__version__)
print(device)

symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
look_up = {s: i for i, s in enumerate(symbols)}
symbols = set(symbols)

def text_to_sequence(text):
    text = text.lower()
    return [look_up[s] for s in text if s in symbols]

text = "Hello world! Text to speech!"
print(text_to_sequence(text))

processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

text = "Hello world! Text to speech!"
processed, lengths = processor(text)

print(processed)
print(lengths)

print([processor.tokens[i] for i in processed[0, : lengths[0]]])
