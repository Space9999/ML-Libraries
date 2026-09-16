import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Transformer.Vocabulary import Vocabulary
from Transformer.Transformer import Transformer
from Transformer import Transformer_Utils

batch_size = 50
n_epochs = 100
training_size = 2000

english_corpus = []
spanish_corpus = []
def load_phrases(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.read().split("\n"):
            splitted = line.split("\t")
            if (len(splitted) == 1):
                continue
            english = splitted[0].strip()
            spanish = splitted[1].strip()
            english_corpus.append(english)
            spanish_corpus.append(spanish)

load_phrases("C:/Users/sbodd/ML From Scratch/Data/spa.txt")
print(english_corpus[5], spanish_corpus[8])
vocab = Vocabulary(english_corpus)
for sentence in spanish_corpus:
    vocab.add_vocab(vocab.tokenize(sentence))

vocab_size = len(list(vocab.get_token2index().keys()))

parallel_corpus = [{"src": english, "tgt": spanish} 
                   for english, spanish in zip(english_corpus[:training_size], spanish_corpus[:training_size])]
batches, masks = Transformer_Utils.get_batches_and_masks(parallel_corpus, vocab, batch_size)
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

        



            
