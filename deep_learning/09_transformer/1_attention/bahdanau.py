import torch
from torch import nn
import hashlib
import os
import zipfile
import tarfile
import requests
import collections
import math
from matplotlib import pyplot
from show_heatmap import show_heatmap
from add_attention import AdditiveAttention

class AttentionDecoder(nn.Module):
    '''The base attention-based decoder interface.'''
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
    
def download(url, folder='../data', sha1_hash=None):
    '''Download a file to folder and return the local filepath.'''
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check is hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1024 * 1024)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def extract(filename, folder=None):
    '''Extract a zip/tar file into folder.'''
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)
    return base_dir

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
        
    @property
    def attention_weights(self):
        return self._attention_weights


astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)

class RNN(nn.Module):
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)


class GRU(RNN):
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        nn.Module.__init__(self)
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)

class Seq2SeqEncoder(nn.Module):
    '''The RNN encoder for sequence-to-sequence learning.'''
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = GRU(embed_size, num_hiddens, num_layers, dropout)
    
    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(astype(transpose(X), torch.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state


class Vocab:
    '''Vocabulary for text.'''
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed.
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
    
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

class MTFraEng():
    '''The English-French dataset.'''
    def _download(self):
        file_path = download('http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip',
                             sha1_hash='94646ad1522d915e7b0f9296181140edcf86a4f5')
        base_dir = extract(file_path)

        with open(os.path.join(base_dir, 'fra-eng/fra.txt'), encoding='utf-8') as f:
            return f.read()
        
    def _preprocess(self, text):
        # Replace non-breaking space with space.
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text.lower())]
        return ''.join(out)
    
    def _tokenize(self, text, max_examples=None):
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples: break
            parts = line.split('\t')
            if len(parts) == 2:
                # Skip empty tokens
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt
        
    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(self._download())
    
    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        def _build_array(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=2)
            array = torch.tensor([vocab[s] for s in sentences])
            valid_len = reduce_sum(astype(array != vocab['<pad>'], torch.int32), 1)
            return array, vocab, valid_len
        src, tgt = self._tokenize(self._preprocess(raw_text),
                                  self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
                src_vocab, tgt_vocab)
    
    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    def build(self, src_sentences, tgt_sentences):
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
            src_sentences, tgt_sentences)])
        arrays, _, _ = self._build_arrays(
            raw_text, self.src_vocab, self.tgt_vocab)
        return arrays
    
class EncoderDecoder(nn.Module):
    '''The base class for the encoder-decoder architecture.'''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
    
    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return nn.functional.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    
    def predict_step(self, batch, num_steps,
                     save_attention_weights=False):
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [expand_dims(tgt[:, 0], 1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], 1), attention_weights
    
class Seq2Seq(EncoderDecoder):
    '''The RNN encoder-decoder for sequence to sequence learning.'''
    def __init__(self, encoder, decoder, tgt_pad, lr):
        self.tgt_pad = tgt_pad
        self.lr = lr
        super().__init__(encoder, decoder)

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        return self.loss(Y_hat, batch[-1])
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Trainer():
    '''The base class for training models with data.'''
    def __init__(self, max_epochs, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        
    def prepare_data(self, data):
        self.train_dataloader = data.get_dataloader(train=True)
        self.val_dataloader = data.get_dataloader(train=False)
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    
    def prepare_model(self, model):
        model.trainer = self
        # model.board.xlim = [0, self.max_epochs]
        self.model = model
    
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        train_his = []
        val_his = []
        for self.epoch in range(self.max_epochs):
            (train_loss, val_loss) = self.fit_epoch()
            train_his.append(train_loss)
            val_his.append(val_loss)
        return (train_his, val_his)

    def fit_epoch(self):
        self.model.train()
        train_loss = []
        val_loss = []
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            train_loss.append(loss.item())
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return (train_loss, val_loss)
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
                val_loss.append(loss.item())
            self.val_batch_idx += 1
    
        return (sum(train_loss) / len(train_loss), sum(val_loss) / len(train_loss))

    def prepare_batch(self, batch):
        return batch
    
    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

def bleu(pred_seq, label_seq, k):
    '''Compute the BLEU.'''
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__ == '__main__':
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    batch_size, num_steps = 4, 7
    
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
    X = torch.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)

    assert output.shape == (batch_size, num_steps, vocab_size)
    assert state[0].shape == (batch_size, num_steps, num_hiddens)
    assert state[1][0].shape == (batch_size, num_hiddens)
    
    data = MTFraEng(batch_size=128)
    embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    
    epochs = 30
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)
    trainer = Trainer(max_epochs=epochs, gradient_clip_val=1)
    (train_his, val_his) = trainer.fit(model, data)
    
    pyplot.plot(range(1, epochs + 1), train_his, label='Train Loss', color='blue', marker='o')
    pyplot.plot(range(1, epochs + 1), val_his, label='Validation Loss', color='red', marker='o')   
    pyplot.legend()
    pyplot.grid()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()    

    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    preds, _ = model.predict_step(
        data.build(engs, fras), data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            translation.append(token)
        print(f'{en} => {translation}, bleu,'
            f'{bleu(" ".join(translation), fr, k=2):.3f}')
        
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [fras[-1]]), data.num_steps, True)
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weights], 0)
    attention_weights = attention_weights.reshape((1, 1, -1, data.num_steps))

    # Plus one to include the end-of-sequence token.
    show_heatmap(attention_weights[:, :, :, :len(engs[-1].split()) + 1].detach().numpy(),
        x_label='Key positions', y_label='Query positions')
