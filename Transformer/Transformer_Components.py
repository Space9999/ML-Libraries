import numpy as np
import math
import copy
from utils.Layers import DenseMultiDim, Dense, Activation, Dropout, LayerNormalization
import utils.Activations_Functions as Activations_Functions
import utils.Base_Neural_Network as NN
import utils.Optimizers as Optimizers

# Fixed optimizer for all used layers
optimizer = Optimizers.Adam()

# References: https://brandonrohrer.com/transformers.html, https://github.com/jsbaan/transformer-from-scratch
class MultiHeadAttention():

    def __init__(self, hidden_dim, num_heads):

        # Should be a integer
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.isSelfAttention = False

        # The 3 comes from the concatenation of the query, key, and value matrices
        self.qkv_projection = DenseMultiDim(input_size = hidden_dim, n_units = 3 * num_heads * self.qkv_dim, have_bias = False)
        self.qkv_projection.initialize_layer(optimizer)

        # For cross attention only
        self.query_projection = DenseMultiDim(input_size = hidden_dim, n_units = num_heads * self.qkv_dim, have_bias = False)
        self.query_projection.initialize_layer(optimizer)
        self.key_value_projection = DenseMultiDim(input_size = hidden_dim, n_units = 2 * num_heads * self.qkv_dim, have_bias = False)
        self.key_value_projection.initialize_layer(optimizer)

        self.output_projection = DenseMultiDim(input_size = hidden_dim, n_units = num_heads * self.qkv_dim, have_bias = False)
        self.output_projection.initialize_layer(optimizer)

    def self_attention_projection(self, input):
        batch_size, sequence_length, _ = input.shape
        qkv = self.qkv_projection.forward_pass(input)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        query, key, value = np.array_split(qkv, 3, axis = -1)

        return query, key, value

    def cross_attention_projection(self, encoder_hidden_states, decoder_hidden_states):
        batch_size, source_sequence_length, _ = encoder_hidden_states.shape
        _, target_sequence_length, _ = decoder_hidden_states.shape

        query = self.query_projection.forward_pass(input = decoder_hidden_states)
        query = query.reshape(batch_size, target_sequence_length, self.num_heads, self.qkv_dim)

        key_value = self.key_value_projection.forward_pass(input = encoder_hidden_states)
        key_value = key_value.reshape(batch_size, source_sequence_length, self.num_heads, 2 * self.qkv_dim)
        key, value = np.array_split(key_value, 2, axis = -1)

        return query, key, value

    def mask_logits(self, logits, source_padding_mask = None, future_mask = None):
        masked_logits = logits
        mask = np.zeros_like(logits, dtype = bool)
        if source_padding_mask is not None:            
            mask |= source_padding_mask[:, np.newaxis, np.newaxis, :] == 0

        if future_mask is not None:
            mask |= future_mask[np.newaxis, np.newaxis, :, :]

        # e^-inf is treated as 0 in python so, these values are disregarded from softmax
        masked_logits[mask] = float("-inf")
        # Save for backward pass
        self.combined_mask = mask

        return masked_logits

    def scaled_dot_product(self, query, key, value, source_padding_mask = None, future_mask = None):
        # Swapping the last two axes ensures that the matrix multiplication works even if query and key positions are not of same size
        attention_logits = np.matmul(query, key.transpose(0, 1, 3, 2))
        attention_logits = attention_logits / np.sqrt(query.shape[-1])

        if source_padding_mask is not None or future_mask is not None:
            attention_logits = self.mask_logits(attention_logits, source_padding_mask, future_mask)

        self.attention = Activations_Functions.softmax(attention_logits)
        attention_value = np.matmul(self.attention, value)

        return attention_value

    def forward_pass(self, input, encoder_hidden_states = None, source_padding_mask = None, future_mask = None):
        self.batch_size, self.sequence_length, self.hidden_dim = input.shape

        # "input" variable should be the decoder hidden states for cross attention
        if encoder_hidden_states is None:
            self.isSelfAttention = True
            self.query, self.key, self.value = self.self_attention_projection(input)
        else:
            self.isSelfAttention = False
            self.encoder_sequence_length = encoder_hidden_states.shape[1]
            self.query, self.key, self.value = self.cross_attention_projection(encoder_hidden_states, input)

        # Swapped dimesions to make matrix multiplication work for scaled dot product
        self.query_transpose = self.query.transpose(0, 2, 1, 3)
        self.key_transpose = self.key.transpose(0, 2, 1, 3)
        self.value_transpose = self.value.transpose(0, 2, 1, 3)

        attention_values = self.scaled_dot_product(self.query_transpose, self.key_transpose, self.value_transpose, source_padding_mask, future_mask)
        attention_values = attention_values.transpose(0, 2, 1, 3).reshape(self.batch_size, self.sequence_length, self.hidden_dim)

        output = self.output_projection.forward_pass(attention_values)
        return output

    def backward_pass(self, accum_grad):
        output_grad = self.output_projection.backward_pass(accum_grad)
        
        attention_value_grad = output_grad.reshape(self.batch_size, self.sequence_length, self.num_heads, self.qkv_dim)
        attention_value_grad = attention_value_grad.transpose(0, 2, 1, 3)

        attention_grad = np.matmul(attention_value_grad, self.value_transpose.transpose(0, 1, 3, 2))
        value_grad = np.matmul(self.attention.transpose(0, 1, 3, 2), attention_value_grad)
        logits_grad = Activations_Functions.softmax_grad(attention_grad, self.attention)

        if self.combined_mask is not None:
            logits_grad[self.combined_mask] = 0

        logits_grad = logits_grad / np.sqrt(self.query_transpose.shape[-1])
        query_grad = np.matmul(logits_grad, self.key_transpose)
        key_grad = np.matmul(logits_grad.transpose(0, 1, 3, 2), self.query_transpose)

        query_grad = query_grad.transpose(0, 2, 1, 3)
        key_grad = key_grad.transpose(0, 2, 1, 3)
        value_grad = value_grad.transpose(0, 2, 1, 3)

        if self.isSelfAttention:
            qkv_grad = np.concatenate([query_grad, key_grad, value_grad], axis = -1)
            qkv_grad = qkv_grad.reshape(self.batch_size, self.sequence_length, 3 * self.num_heads * self.qkv_dim)
            input_grad = self.qkv_projection.backward_pass(qkv_grad)
            return input_grad
        
        query_grad = query_grad.reshape(self.batch_size, self.sequence_length, self.num_heads * self.qkv_dim)
        decoder_hidden_grad = self.query_projection.backward_pass(query_grad)

        key_value_grad = np.concatenate([key_grad, value_grad], axis = -1)
        key_value_grad = key_value_grad.reshape(self.batch_size, self.encoder_sequence_length, 2 * self.num_heads * self.qkv_dim)
        encoder_hidden_grad = self.key_value_projection.backward_pass(key_value_grad)

        return encoder_hidden_grad, decoder_hidden_grad

# Specific reference for math: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
class PositionalEncoding():

    # All calculations are done in init method for precomputation
    def __init__(self, hidden_dim, max_length = 5000):
        self.positional_embedding = np.zeros((max_length, hidden_dim))
        position = np.arange(0, max_length, dtype = float)
        position = position[:, np.newaxis]

        # This is mathematically equivalent to the formula shown in the reference
        # Natural log and exponent is more efficient than direct division for vector embeddings
        division_term = np.exp(np.arange(0, hidden_dim, 2, dtype = float) * (-math.log(10000.0) / hidden_dim))
        self.positional_embedding[:, 0::2] = np.sin(position * division_term)
        self.positional_embedding[:, 1::2] = np.cos(position * division_term)

        self.positional_embedding = self.positional_embedding[np.newaxis, :]

    def forward_pass(self, embeddings):
        # Adds positional embeddings up to the sequence size 
        return embeddings + self.positional_embedding[:, :embeddings.shape[1]]

class EncoderBlock():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, dropout_probability):
        self.mha = MultiHeadAttention(hidden_dim, num_heads)

        self.feedforward = NN.Base_Neural_Network(optimizer = optimizer)
        self.feedforward.add(DenseMultiDim(input_size = hidden_dim, n_units = feedforward_dim))
        self.feedforward.add(Activation("relu"))
        self.feedforward.add(DenseMultiDim(input_size = feedforward_dim, n_units = hidden_dim))

        self.dropout1 = Dropout(dropout_probability)
        self.dropout2 = Dropout(dropout_probability)
        self.layer_norm1 = LayerNormalization(input_size = hidden_dim)
        self.layer_norm2 = LayerNormalization(input_size = hidden_dim)
        self.layer_norm1.initialize_layer(optimizer)
        self.layer_norm2.initialize_layer(optimizer)

    def forward_pass(self, input, source_padding_mask):
        mha_output = self.mha.forward_pass(input, source_padding_mask = source_padding_mask)
        block_output = self.dropout1.forward_pass(mha_output)
        res_connection1 = input + block_output
        block_output_res = self.layer_norm1.forward_pass(res_connection1)

        ff_output = self.feedforward.forward_pass(block_output_res)
        block_output2 = self.dropout2.forward_pass(ff_output)
        res_connection2 = block_output_res + block_output2
        block_output_res2 = self.layer_norm2.forward_pass(res_connection2)

        return block_output_res2

    def backward_pass(self, accum_grad):
        res_connection2_grad = self.layer_norm2.backward_pass(accum_grad)
        block_output_res_grad = res_connection2_grad
        block_output2_grad = res_connection2_grad

        ff_output_grad = self.dropout2.backward_pass(block_output2_grad)
        block_output_res_ff_grad = self.feedforward.backward_pass(ff_output_grad)
        res_connection1_grad = self.layer_norm1.backward_pass(block_output_res_grad + block_output_res_ff_grad)

        input_res_grad = res_connection1_grad
        block_output_grad = res_connection1_grad
        mha_output_grad = self.dropout1.backward_pass(block_output_grad)
        input_mha_grad = self.mha.backward_pass(mha_output_grad)

        return input_res_grad + input_mha_grad

class DecoderBlock():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, dropout_probability):
        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)

        self.feedforward = NN.Base_Neural_Network(optimizer = optimizer)
        self.feedforward.add(DenseMultiDim(input_size = hidden_dim, n_units = feedforward_dim))
        self.feedforward.add(Activation("relu"))
        self.feedforward.add(DenseMultiDim(input_size = feedforward_dim, n_units = hidden_dim))

        self.dropout1 = Dropout(dropout_probability)
        self.dropout2 = Dropout(dropout_probability)
        self.dropout3 = Dropout(dropout_probability)
        self.layer_norm1 = LayerNormalization(input_size = hidden_dim)
        self.layer_norm2 = LayerNormalization(input_size = hidden_dim)
        self.layer_norm3 = LayerNormalization(input_size = hidden_dim)
        self.layer_norm1.initialize_layer(optimizer)
        self.layer_norm2.initialize_layer(optimizer)
        self.layer_norm3.initialize_layer(optimizer)

    def forward_pass(self, input, encoder_hidden_states, source_padding_mask, future_mask):
        self_mha_output = self.self_mha.forward_pass(input, future_mask = future_mask)
        block_output = self.dropout1.forward_pass(self_mha_output)
        res_connection1 = input + block_output
        block_output_res = self.layer_norm1.forward_pass(res_connection1)

        cross_mha_output = self.cross_mha.forward_pass(block_output_res, encoder_hidden_states, source_padding_mask)
        block_output2 = self.dropout2.forward_pass(cross_mha_output)
        res_connection2 = block_output_res + block_output2
        block_output_res2 = self.layer_norm2.forward_pass(res_connection2)

        ff_output = self.feedforward.forward_pass(block_output_res2)
        block_output3 = self.dropout3.forward_pass(ff_output)
        res_connection3 = block_output_res2 + block_output3
        block_output_res3 = self.layer_norm3.forward_pass(res_connection3)
        
        return block_output_res3

    def backward_pass(self, accum_grad):
        res_connection3_grad = self.layer_norm3.backward_pass(accum_grad)
        block_output3_grad = res_connection3_grad
        block_output_res2_grad = res_connection3_grad
        ff_output_grad = self.dropout3.backward_pass(block_output3_grad)
        block_output_res2_ff_grad = self.feedforward.backward_pass(ff_output_grad)

        res_connection2_grad = self.layer_norm2.backward_pass(block_output_res2_grad + block_output_res2_ff_grad)
        block_output_res_grad = res_connection2_grad
        block_output2_grad = res_connection2_grad
        cross_mha_output_grad = self.dropout2.backward_pass(block_output2_grad)
        encoder_hidden_grad, block_output_res_mha_grad = self.cross_mha.backward_pass(cross_mha_output_grad)

        res_connection1_grad = self.layer_norm1.backward_pass(block_output_res_grad + block_output_res_mha_grad)
        input_grad = res_connection1_grad
        block_output_grad = res_connection1_grad
        self_mha_output_grad = self.dropout1.backward_pass(block_output_grad)
        mha_input_grad = self.self_mha.backward_pass(self_mha_output_grad)

        return mha_input_grad + input_grad, encoder_hidden_grad

class Encoder():

    def __init__(self, hidden_dim, feedforward_dim, embedding, num_heads, num_blocks, dropout_probability):
        self.embedding = copy.copy(embedding)
        self.embedding.initialize_layer(optimizer = optimizer)
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.dropout = Dropout(dropout_probability)
        self.encoder_blocks = []
        for _ in range(num_blocks):
            self.encoder_blocks.append(EncoderBlock(hidden_dim, feedforward_dim, num_heads, dropout_probability))
        
    def forward_pass(self, input_ids, source_padding_mask):
        output = self.embedding.forward_pass(input_ids) * math.sqrt(self.hidden_dim)
        output = self.positional_encoding.forward_pass(output)
        output = self.dropout.forward_pass(output)

        for encoder_block in self.encoder_blocks:
            output = encoder_block.forward_pass(output, source_padding_mask)

        return output

    def backward_pass(self, accum_grad):
        output_grad = accum_grad
        for encoder_block in reversed(self.encoder_blocks):
            output_grad = encoder_block.backward_pass(output_grad)

        output_grad = self.dropout.backward_pass(output_grad)
        output_grad *= math.sqrt(self.hidden_dim)
        output_grad = self.embedding.backward_pass(output_grad)
        # No learnable parameters in positional encoding
        return output_grad

class Decoder():
    def __init__(self, hidden_dim, feedforward_dim, embedding, vocab_size, num_heads, num_blocks, dropout_probability):
        self.embedding = copy.copy(embedding)
        self.embedding.initialize_layer(optimizer = optimizer)
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.dropout = Dropout(dropout_probability)
        self.decoder_blocks = []
        for _ in range(num_blocks):
            self.decoder_blocks.append(DecoderBlock(hidden_dim, feedforward_dim, num_heads, dropout_probability))

        self.output_layer = DenseMultiDim(vocab_size, hidden_dim, have_bias = False)
        self.output_layer.initialize_layer(optimizer)
            
    def forward_pass(self, input_tokens, encoder_hidden_states, source_padding_mask, future_mask):
        output = self.embedding.forward_pass(input_tokens) * math.sqrt(self.hidden_dim)
        output = self.positional_encoding.forward_pass(output)
        output = self.dropout.forward_pass(output)

        for decoder_block in self.decoder_blocks:
            output = decoder_block.forward_pass(output, encoder_hidden_states, source_padding_mask, future_mask)

        output = self.output_layer.forward_pass(output)

        # Save for backward pass
        self.encoder_hidden_shape = encoder_hidden_states.shape

        return output

    def backward_pass(self, accum_grad):
        output_grad = accum_grad
        output_grad = self.output_layer.backward_pass(output_grad)
        encoder_hidden_total_grad = np.zeros(self.encoder_hidden_shape)

        for decoder_block in reversed(self.decoder_blocks):
            output_grad, encoder_hidden_grad = decoder_block.backward_pass(output_grad)
            encoder_hidden_total_grad += encoder_hidden_grad

        output_grad = self.dropout.backward_pass(output_grad)
        output_grad *= math.sqrt(self.hidden_dim)
        output_grad = self.embedding.backward_pass(output_grad)
        # No learnable parameters in positional encoding
        return output_grad, encoder_hidden_total_grad







            

        

        

        


