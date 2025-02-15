
from matplotlib import pyplot
import tensorflow as tf
import tensorflow_text
import tensorflow_datasets as tfds
import keras
import numpy

MAX_TOKENS = 128

def positional_encoding(length, depth):
    depth = depth / 2

    positions = numpy.arange(length)[:, numpy.newaxis]  # (seq, 1)
    depths = numpy.arange(depth)[numpy.newaxis, :] / depth # (1, depth)

    angle_rates = 1 / (10000**depths)   # (1, depth)
    angle_rads = positions * angle_rates # (pos, depth)

    pos_encoding = numpy.concatenate(
        [numpy.sin(angle_rads), numpy.cos(angle_rads)],
        axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative savel of the embedding and position_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class BaseAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(query=x,
                                            key=context,
                                            value=context,
                                            return_attention_scores=True)
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x,
                               value=x,
                               key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x,
                               value=x,
                               key=x,
                               use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()
    
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

class EncoderLayer(keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
    
        self.ffn = FeedForward(d_model, dff)
    
    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(keras.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        # x is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)   # shape (batch_size, seq_len, d_model)

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        
        return x    # shape (batch_size, seq_len, d_model)
    
class DecoderLayer(keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)
    
    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later.
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)     # shape (batch_size, seq_len, d_model)
        return x

class Decoder(keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        # x is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)   # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

class Transformer(keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_heads, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)
        self.final_layer = keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs):
        # To use a Keras model with .fit you must pass all your inputs in the first argument.
        context, x = inputs
        context = self.encoder(context) # (batch_size, context_len, d_model)
        x = self.decoder(x, context)    # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)    # (batch_size, target_len, target_vocab_size)

        # Return the final output and the attention weights.
        return logits

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is English, initialize the output with the
    # English `[START]` token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return text, tokens, attention_weights

if __name__ == '__main__':
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('> Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print('> Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    model_name = 'ted_hrlr_translate_pt_en_converter'
    keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True)
    tokenizers = tf.saved_model.load('ted_hrlr_translate_pt_en_converter_extracted/' + model_name)

    print('> This is a batch of strings:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

    encoded = tokenizers.en.tokenize(en_examples)
    print('> This is a padded-batch of token IDs:')
    for row in encoded.to_list():
        print(row)

    round_trip = tokenizers.en.detokenize(encoded)
    print('> This is human-readable text:')
    for line in round_trip.numpy():
        print(line.decode('utf-8'))

    print('> This is the text split into tokens:')
    tokens = tokenizers.en.lookup(encoded)
    print(tokens)

    lengths = []
    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())
    
        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)
    print()

    all_lengths = numpy.concatenate(lengths)
    pyplot.hist(all_lengths, numpy.linspace(0, 500, 101))
    pyplot.ylim(pyplot.ylim())
    max_length = max(all_lengths)
    pyplot.plot([max_length, max_length], pyplot.ylim())
    pyplot.title(f'Maximum tokens per examples: {max_length}')
    pyplot.show()

    def prepare_batch(pt, en):
        pt = tokenizers.pt.tokenize(pt) # output is ragged
        pt = pt[:, :MAX_TOKENS]    # trim to MAX_TOKENS
        pt = pt.to_tensor() # convert to 0-padded dense tensor

        en = tokenizers.en.tokenize(en)
        en = en[:, :(MAX_TOKENS + 1)]
        en_inputs = en[:, :-1].to_tensor() # drop the [END] tokens
        en_labels = en[:, 1:].to_tensor() # drop the [START] tokens

        return (pt, en_inputs), en_labels

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64

    def make_batches(ds):
        return (ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(
            prepare_batch, tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE))
    
    # Create training and validation set batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    for (pt, en), en_labels in train_batches.take(1):
        break

    print(pt.shape)
    print(en.shape)
    print(en_labels.shape)
    print(en[0][:10])
    print(en_labels[0][:10])

    pos_encoding = positional_encoding(length=2048, depth=512)
    # Check the shape.
    print(pos_encoding.shape)

    # Plot the dimensions.
    pyplot.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    pyplot.ylabel('Depth')
    pyplot.xlabel('Position')
    pyplot.colorbar()
    pyplot.show()

    embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size().numpy(), d_model=512)
    embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size().numpy(), d_model=512)

    pt_emb = embed_pt(pt)
    en_emb = embed_en(en)

    d = {'color': 'blue', 'age': 22, 'type': 'pickup'}
    result = d['color']
    print(result)

    sample_ca = CrossAttention(num_heads=2, key_dim=512)
    print('> Cross attention:')
    print(pt_emb.shape)
    print(en_emb.shape)
    print(sample_ca(en_emb, pt_emb).shape)

    sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)

    print(pt_emb.shape)
    print(sample_gsa(pt_emb).shape)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
                              target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                              dropout_rate=dropout_rate)

    output = transformer((pt, en))
    print(en.shape)
    print(pt.shape)
    print(output.shape)

    # transformer.summary()

    class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super().__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, dtype=tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                      epsilon=1e-9)
    
    def masked_loss(label, pred):
        mask = label != 0
        loss_object = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = loss_object(label, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
        return loss

    def masked_accuracy(label, pred):
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, pred.dtype)
        match = label == pred

        mask = label != 0

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match)/tf.reduce_sum(mask)
    
    transformer.compile(loss=masked_loss,
                        optimizer=optimizer,
                        metrics=[masked_accuracy])
    
    transformer.fit(train_batches,
                    epochs=20,
                    validation_data=val_batches)
    
    translator = Translator(tokenizers, transformer)

    def print_translation(sentence, tokens, ground_truth):
        print(f'{"Input:":15s}: {sentence}')
        print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
        print(f'{"Ground truth":15s}: {ground_truth}')
    
    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'
    translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)

    sentence = 'os meus vizinhos ouviram sobre esta ideia.'
    ground_truth = 'and my neighboring homes heard about this idea .'
    translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)

    sentence = 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.'
    ground_truth = "so i'll just share with you some stories very quickly of some magical things that have happened."
    translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)
