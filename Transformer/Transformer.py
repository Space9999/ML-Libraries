from Transformer.Transformer_Components import Encoder, Decoder
from utils.Layers import Embedding
from utils.Loss_Functions import sparse_categorical_cross_entropy, sparse_categorical_cross_entropy_grad
from utils.Activations_Functions import softmax, softmax_grad
import numpy as np

class Transformer():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, num_blocks, 
                 vocab_size, padding_index, dropout_probability):
        embedding = Embedding(vocab_size, hidden_dim)
        self.encoder = Encoder(hidden_dim, feedforward_dim, embedding, num_heads, num_blocks, dropout_probability)
        self.decoder = Decoder(hidden_dim, feedforward_dim, embedding, vocab_size, num_heads, num_blocks, 
                               dropout_probability)
        self.padding_index = padding_index
        self.hidden_dim = hidden_dim

    def train(self, epochs, batches, masks):
        for epoch in range(epochs):
            # Iterates through rows of batches appended with appropriate masks
            for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
                zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
            ):
                encoder_output = self.encoder.forward_pass(src_batch, src_mask)
                decoder_output_original = self.decoder.forward_pass(tgt_batch, encoder_output,
                                                                    src_mask, tgt_mask)
                # Last decoder output is meaningless as it does not have a target token
                decoder_output = decoder_output_original[:, :-1, :]
                # The BOS token should not be included in loss
                tgt_batch = tgt_batch[:, 1:]

                # Ignore padding values in loss and accuracy using mask
                valid_mask = tgt_batch != self.padding_index
                correct_tokens = decoder_output.argmax(axis = -1) == tgt_batch
                
                batch_loss = sparse_categorical_cross_entropy(tgt_batch, decoder_output, valid_mask)
                batch_accuracy = (np.sum(valid_mask & correct_tokens)) / np.sum(valid_mask)

                # Backward pass portion starts here
                valid_mask = valid_mask[:, :, np.newaxis]
                batch_loss_grad = sparse_categorical_cross_entropy_grad(tgt_batch, decoder_output, valid_mask)
                decoder_output_grad = batch_loss_grad

                # Restores original shapes and fills removed values with 0's
                decoder_original_grad = np.zeros_like(decoder_output_original)
                decoder_original_grad[:, :-1, :] = decoder_output_grad

                _, encoder_grad = self.decoder.backward_pass(decoder_original_grad)
                _ = self.encoder.backward_pass(encoder_grad)

                print(
                    f"epoch: {epoch}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )

        # Return most recent batch loss and accuracy
        return batch_loss, batch_accuracy

                

                
                

                


                



                

                


