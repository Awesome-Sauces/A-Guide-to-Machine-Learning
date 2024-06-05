# Lesson 2 CONTINUED
Let's continue exploring the components and workings of transformer models and unsupervised learning in more detail. We'll delve into each key concept with explanations and accompanying code snippets.

## Unsupervised Learning
Unsupervised learning involves finding patterns or structures in data without using labeled outcomes. Common tasks include clustering, dimensionality reduction, and generative modeling.

*Example: Clustering with K-Means*

```python
import torch

# Generate some random data
data = torch.randn(100, 2)

# Initialize centroids randomly
k = 3
centroids = data[torch.randperm(data.size(0))[:k]]

for _ in range(10):  # Iterate 10 times
    # Assign clusters
    distances = torch.cdist(data, centroids)
    cluster_assignments = distances.argmin(dim=1)

    # Update centroids
    new_centroids = torch.stack([data[cluster_assignments == i].mean(dim=0) for i in range(k)])
    if torch.all(centroids == new_centroids):
        break
    centroids = new_centroids

print("Cluster centroids:", centroids)
```

## Transformers
Transformers are the core architecture used in many state-of-the-art NLP models. They use attention mechanisms to process sequential data.

Basic Transformer Components:

### Embedding Layer
Transforms tokens into dense vectors.

```python
import torch.nn as nn

vocab_size = 10000
embed_dim = 512
embedding = nn.Embedding(vocab_size, embed_dim)

# Example input
input_tokens = torch.randint(0, vocab_size, (32, 100))  # Batch of 32 sequences, each of length 100
embedded_tokens = embedding(input_tokens)
print(embedded_tokens.shape)  # Output shape: (32, 100, 512)
```

### Positional Encoding
Adds positional information to embeddings.

```python
import torch
import math

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
embed_dim = 512
pos_encoder = PositionalEncoding(embed_dim)
pos_encoded_tokens = pos_encoder(embedded_tokens)
print(pos_encoded_tokens.shape)  # Output shape: (32, 100, 512)
```

### Self-Attention Mechanism
Computes attention scores and outputs.

```python
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
num_heads = 8
self_attention = SelfAttention(embed_dim, num_heads)
attention_output = self_attention(pos_encoded_tokens)
print(attention_output.shape)  # Output shape: (32, 100, 512)
```

## Transformer Encoder Layer
Combines self-attention and feedforward layers.

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads)
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
encoded_tokens = encoder_layer(attention_output)
print(encoded_tokens.shape)  # Output shape: (32, 100, 512)
```

## Transformer Model
Combines multiple encoder layers into a full transformer model.

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

## Training Loop
Define a training loop to optimize the model parameters.

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

## Text Generation
Generate text using the trained model.

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
By understanding each component of a transformer model and how they come together, you can build and train powerful models for various NLP tasks. Here’s a quick recap of what we covered:

- Unsupervised Learning: Techniques like clustering to find patterns in data.
- Transformers: An architecture that uses self-attention to handle sequential data.
- Embedding and Positional Encoding: Convert tokens to dense vectors and add positional information.
- Self-Attention: Compute relationships between tokens.
- Encoder Layers: Combine self-attention and feedforward layers.
- Full Transformer Model: Stack encoder layers to build a transformer.
- Training Loop: Optimize the model parameters using backpropagation.
- Text Generation: Use the trained model to generate text.
