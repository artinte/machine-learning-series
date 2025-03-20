import torch

# In PyTorch 2.6, we changed the default value of the `weights_only` argument
# in `torch.load` from `False` to `True`.
torch_load = torch.load

def torch_load_patched(*args, **kwargs):
    kwargs["weights_only"] = False  # 解决 UnpicklingError
    return torch_load(*args, **kwargs)

torch.load = torch_load_patched

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
#  CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 8.00 GiB of
# which 0 bytes is free. Of the allocated memory 14.51 GiB is allocated by PyTorch,
# and 15.90 MiB is reserved by PyTorch but unallocated.
# preload_models()

# 加载低显存，CUDA 只有 8 G
# text.pt 2.32G
# coarse.pt 1.25G
# fine.pt 1.11G
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav('bark_generation.wav', SAMPLE_RATE, audio_array)
