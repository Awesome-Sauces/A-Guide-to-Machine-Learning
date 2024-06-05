# Building and Training a Transformer Model in PyTorch

### Step 1: Data Preparation
Load and preprocess the Shakespeare text data. We'll convert the text to lowercase, tokenize it, and create sequences for training.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import re

# Load Shakespeare text from file
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    shakespeare_text = file.read()

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.lower()
    return text

shakespeare_text = preprocess_text(shakespeare_text)

# Tokenize the text
def tokenize(text):
    chars = list(set(text))
    token_to_id = {char: i for i, char in enumerate(chars)}
    id_to_token = {i: char for i, char in enumerate(chars)}
    token_ids = [token_to_id[char] for char in text]
    return token_ids, token_to_id, id_to_token

token_ids, token_to_id, id_to_token = tokenize(shakespeare_text)

# Define the dataset and data loader
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_length]), 
                torch.tensor(self.data[idx + 1:idx + self.seq_length + 1]))

seq_length = 100
batch_size = 64
dataset = TextDataset(token_ids, seq_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### Step 2: Model Definition
Define the transformer model, including the self-attention mechanism, positional encoding, and transformer encoder layers.

```python
import math

# Self-Attention Mechanism
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

# Multi-Head Attention Mechanism
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

# Positional Encoding
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

# Transformer Encoder Layer
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

# Complete Transformer Model
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
```

### Step 4: Training the Model
Define the loss function and optimizer, and implement the training loop.

```python
device = torch.device('cpu')  # Change to 'cuda' if you have a GPU

vocab_size = len(token_to_id)
embed_dim = 256
num_heads = 8
hidden_dim = 512
num_layers = 6
model = TransformerModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)

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

### Step 5: Text Generation
Implement a function to generate text using the trained model.

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
- Data Preparation: Load and preprocess the Shakespeare dataset, tokenize it, and create sequences for training.
- Model Definition: Define the self-attention mechanism, multi-head attention, positional encoding, and transformer encoder layers.
- Training the Model: Define the loss function and optimizer, and implement the training loop to optimize the model parameters.
- Text Generation: Use the trained model to generate text by predicting the next token iteratively.
