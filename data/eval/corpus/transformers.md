# Transformers and Attention

The transformer architecture, introduced in the 2017 paper "Attention Is All You Need", replaces recurrence with self-attention. In self-attention every token produces a query, a key and a value vector; the attention weight between two tokens is the scaled dot product of the query and the key, normalised with a softmax, and the output is the weighted sum of the values.

Multi-head attention runs several attention functions in parallel with different learned projections, allowing the model to attend to information from different representation subspaces. Because attention has no notion of order, positional encodings (sinusoidal or learned) are added to the token embeddings.

The original transformer is an encoder-decoder model. The encoder maps the input sequence to contextual representations; the decoder generates the output one token at a time, attending to both the previously generated tokens (masked self-attention) and the encoder output (cross-attention). BERT uses only the encoder and is pre-trained with masked language modelling; GPT-style models use only the decoder and are trained with next-token prediction.

The computational cost of self-attention grows quadratically with sequence length, which motivates efficient variants such as sparse attention, linear attention and sliding-window attention.
