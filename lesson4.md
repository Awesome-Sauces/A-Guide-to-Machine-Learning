# Attention is all you need!
The "Attention Is All You Need" paper by Vaswani et al. (2017) introduced the transformer architecture, which revolutionized natural language processing (NLP) and many other areas of machine learning. The transformer model relies solely on attention mechanisms to handle dependencies between input and output tokens, dispensing with recurrence and convolutions entirely.

Let's break down the key concepts and contributions of the paper step-by-step.

## Key Concepts from the Paper

### Self-Attention Mechanism
The self-attention mechanism allows the model to weigh the importance of different tokens in a sequence relative to each other. This is in contrast to traditional RNNs and CNNs that process tokens sequentially or with fixed-size windows, respectively.

Steps in Self-Attention:

- Input Embeddings: Convert tokens to dense vectors
- Query, Key, Value Vectors: For each token, create three vectors (Query Q, Key K, Value V) by multiplying the input embedding by learned weight matrices W^Q, W^K, W^V
- Attention Scores: Computer scores by taking the dot produc tof Q and K, and scale them by the square root of the dimension of the key vectors (root(d_k)).
- Softmax: Apply the softmax function to the scores to get attention weights.
- Weighted Sum: Multiply the attention weights by the value vectors V and sum them to get the output for each token.

### Multi-Head Attention
Multi-head attention allows the model to focus on different parts of the sequence simultaneously by using multiple sets of Query, Key, and Value matrices. This enables the model to capture various aspects of the token relationships.

Steps in Multi-Head Attention:

- Linear Projections: Linearly project the queries, keys, and values h times with different learned linear projections.
- Parallel Attention: Apply self-attention mechanism in parallel for each head.
- Concatenate: Concatenate the outputs of each head.
- Final Linear Projection: Apply a final linear projection to the concatenated outputs.

### Positional Encoding
Since transformers process tokens in parallel, they need a way to incorporate the order of tokens. Positional encoding adds positional information to the token embeddings so the model can learn the importance of the order.

### Encoder-Decoder Architecture
The original transformer model consists of an encoder and a decoder, each composed of stacked layers.

Encoder:

- Input Embedding: Converts input tokens to embeddings.
- Positional Encoding: Adds positional information to embeddings.
- N Encoder Layers: Each layer has two sub-layers: multi-head self-attention and position-wise feedforward network.

Decoder:

- Target Embedding: Converts target tokens to embeddings.
- Positional Encoding: Adds positional information to embeddings.
- N Decoder Layers: Each layer has three sub-layers: multi-head self-attention, multi-head attention over encoder outputs, and position-wise feedforward network.

### Position-wise Feedforward Networks
Each position in the sequence is processed independently and identically by a fully connected feedforward network, which consists of two linear transformations with a ReLU activation in between.

### Layer Normalization and Residual Connections
Layer normalization is applied after each sub-layer (attention and feedforward), and residual connections are added to ensure stable gradients.

## Low-Level Understanding
Let's dive into some code snippets to see how these components are implemented.

### Self-Attention Mechanism
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Transform inputs
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Compute attention output
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(output)

# Example usage
embed_dim = 512
num_heads = 8
self_attention = SelfAttention(embed_dim, num_heads)
x = torch.randn(32, 100, embed_dim)  # Batch of 32 sequences, each of length 100
attention_output = self_attention(x)
print(attention_output.shape)  # Output shape: (32, 100, 512)
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.fc(attention_output)
        return output

# Example usage
multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
attention_output = multi_head_attention(x)
print(attention_output.shape)  # Output shape: (32, 100, 512)
```

### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]

# Example usage
pos_encoder = PositionalEncoding(embed_dim)
pos_encoded_tokens = pos_encoder(x)
print(pos_encoded_tokens.shape)  # Output shape: (32, 100, 512)
```

### Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention_output = self.self_attention(x)
        x = self.layernorm1(x + attention_output)
        feedforward_output = self.feedforward(x)
        x = self.layernorm2(x + feedforward_output)
        return x

# Example usage
hidden_dim = 2048
encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
encoded_tokens = encoder_layer(pos_encoded_tokens)
print(encoded_tokens.shape)  # Output shape: (32, 100, 512)
```

### Complete Transformer Model
Combining all the components to form the complete transformer model.

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.fc(x)
        return x

# Example usage
num_layers = 6
model = TransformerModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)
input_tokens = torch.randint(0, vocab_size, (32, 100))  # Batch of 32 sequences, each of length 100
output_tokens = model(input_tokens)
print(output_tokens.shape)  # Output shape: (32, 100, vocab_size)
```

### Training and Text Generation
Finally, let's implement the training loop and text generation functions.

Training Loop
```python
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for batch, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch}, Loss: {loss.item()}')
```

Text Generation
```python
def generate_text(model, start_str, length):
    model.eval()
    input_ids = torch.tensor([token_to_id[char] for char in start_str], dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_str
    for _ in range(length):
        with torch.no_grad():
            output = model(input_ids)
            next_token_id = torch.argmax(output[0, -1, :]).item()
            next_token = id_to_token[next_token_id]
            generated_text += next_token
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
    return generated_text

# Generate some text
start_str = "to be or not to be"
generated_text = generate_text(model, start_str, 500)
print("Generated Text:")
print(generated_text)
```

## Summary
- Self-Attention Mechanism: Computes the relevance of each token to every other token in the sequence.
- Multi-Head Attention: Enhances the self-attention mechanism by using multiple heads to capture different aspects of relationships between tokens.
- Positional Encoding: Adds information about the position of tokens in the sequence.
- Encoder Layer: Combines self-attention and feedforward networks, with residual connections and layer normalization.
- Transformer Model: Stacks multiple encoder layers to form the complete model.
- Training and Text Generation: Optimizes the model and generates text using the trained model.
