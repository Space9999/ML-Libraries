import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Transformer.Vocabulary import Vocabulary
from Transformer.Transformer import Transformer
from Transformer import Transformer_Utils
from random import choices

synthetic_corpus_size = 400
batch_size = 50
n_epochs = 100
n_tokens_in_batch = 5

corpus = ["These are the tokens that will end up in our vocabulary"]
vocab = Vocabulary(corpus)
token2index_keys = list(vocab.get_token2index().keys())
vocab_size = len(token2index_keys)
valid_tokens = token2index_keys[3:]
# Adds random batches of words from token mapping
corpus += [
    " ".join(choices(valid_tokens, k = n_tokens_in_batch))
    for _ in range(synthetic_corpus_size)
]
# Setup a copy task for the transformer as a basic check for transformer mechanics
# i.e if the transformer cannot effectively create the same sentence from source to target, then something is clearly wrong
corpus = [{"src": sent, "tgt": sent} for sent in corpus]
batches, masks = Transformer_Utils.get_batches_and_masks(corpus, vocab, batch_size)
transformer = Transformer(
    # hidden_dim has to be a even number
    hidden_dim = 512,
    feedforward_dim = 2048,
    num_heads = 8,
    num_blocks = 2,
    vocab_size = vocab_size,
    padding_index = vocab.get_token2index()[vocab.PAD],
    dropout_probability = 0.1
)
latest_batch_loss, latest_batch_accuracy = transformer.train(n_epochs, batches, masks)
print(latest_batch_loss, latest_batch_accuracy)