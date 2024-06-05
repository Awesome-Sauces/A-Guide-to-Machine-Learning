# Key Concepts and Terminology

## Unsupervised Learning
Unsupervised learning is a type of machine learning where the model is trained on data without explicit labels. Instead of learning from labeled input-output pairs, the model identifies patterns and structures in the input data. Common unsupervised learning tasks include clustering, dimensionality reduction, and generative modeling.

*Example: Clustering articles based on their topics without knowing the topics beforehand.*

## Transformers
Transformers are a type of neural network architecture designed for handling sequential data, such as text. They have become the backbone of modern natural language processing (NLP) models, including LLMs like GPT-3.

### Key Components:
- Embedding Layer: Converts words or tokens into dense vectors of fixed size.
- Positional Encoding: Adds information about the position of tokens in the sequence, since transformers process the entire sequence simultaneously.
- Self-Attention Mechanism: Computes the importance of each token in the sequence relative to every other token. This allows the model to focus on relevant parts of the sequence.
- Feedforward Layers: Applied after the attention mechanism to process the information further.
- Layer Normalization and Dropout: Techniques to stabilize and regularize the training process.

## Self-Attention
Self-attention is a mechanism that allows each token in a sequence to attend to (or focus on) every other token. This is achieved by computing a weighted sum of the representations of all tokens, where the weights are determined by the relevance of one token to another.

### Steps:
- Query, Key, and Value Vectors: For each token, create three vectors (Query, Key, Value) by multiplying the token embedding with learned weight matrices.
- Attention Scores: Compute attention scores by taking the dot product of the Query vector of one token with the Key vectors of all tokens.
- Softmax: Apply the softmax function to the attention scores to get the attention weights.
- Weighted Sum: Multiply the attention weights by the Value vectors and sum them to get the output for each token.

## Multi-Head Attention
Multi-head attention extends the self-attention mechanism by using multiple sets of Query, Key, and Value weight matrices. This allows the model to capture different aspects of the relationships between tokens.

## Positional Encoding
Since transformers process all tokens in parallel, they need a way to incorporate the order of tokens. Positional encoding adds positional information to the token embeddings, allowing the model to learn the importance of the token order.

## Feedforward Neural Networks
After the attention mechanism, the output is passed through a feedforward neural network for further processing. This typically consists of a couple of fully connected layers with non-linear activation functions.

## Encoder-Decoder Architecture
The original transformer architecture proposed by Vaswani et al. (2017) consists of an encoder and a decoder:
- Encoder: Processes the input sequence and produces a sequence of hidden states.
- Decoder: Takes the hidden states and generates the output sequence, typically for tasks like machine translation.

However, many LLMs, like GPT, use only the encoder part or a variant of it.
