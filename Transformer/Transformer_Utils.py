import numpy as np

# Returns a boolean mask determining if a given token is "ahead" of time
def get_future_mask(sequence_len):
    future_mask = np.triu(np.full((sequence_len, sequence_len), 1), k = 1)
    return future_mask == 1

# Returns batches and masks seperated by source and target
def get_batches_and_masks(corpus, vocab, batch_size):
    pad_token_id = vocab.get_token2index()[vocab.PAD]
    batches = {"src": [], "tgt": []}
    masks = {"src": [], "tgt": []}
    for i in range(0, len(corpus), batch_size):
        src_batch = vocab.batch_encode(
            [pair["src"] for pair in corpus[i : i + batch_size]],
            add_special_tokens = True
        )
        tgt_batch = vocab.batch_encode(
            [pair["tgt"] for pair in corpus[i : i + batch_size]],
            add_special_tokens = True
        )
        src_padding_mask = src_batch != pad_token_id
        future_mask = get_future_mask(tgt_batch.shape[-1])

        batches["src"].append(src_batch)
        batches["tgt"].append(tgt_batch)
        masks["src"].append(src_padding_mask)
        masks["tgt"].append(future_mask)
        
    return batches, masks
