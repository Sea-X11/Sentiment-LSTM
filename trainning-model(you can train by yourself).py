import pandas as pd
import numpy as np
import os
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Set GPU device if available
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# Load and prepare the dataset
dataset_path = "IMDB Dataset.csv"
df = pd.read_csv(dataset_path, sep=",")
print(f"Dataset shape: {df.shape}")
print(f"Sample review: {df.sample()['review'].values[0][:100]}...")

X = df['review'].values
y = df['sentiment'].values
X_traindata, X_testdata, y_traindata, y_testdata = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(f"Training samples: {len(X_traindata)}, Testing samples: {len(X_testdata)}")

# Text preprocessing functions
def preprocess_string(s):
    """Clean and normalize text by removing non-alphabetic characters and converting to lowercase"""
    s = re.sub(r"[^a-zA-Z\s]", '', s)  # Keep letters and spaces
    s = re.sub(r"\s+", ' ', s)  # Normalize whitespace
    return s.strip().lower()  # Trim and convert to lowercase

# Padding function to make sequence lengths uniform
def pad(sentences, seq_len):
    """Pad or truncate sequences to specified length"""
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) > 0:
            features[ii, -min(len(review), seq_len):] = review[:seq_len]
    return features

# Tokenizer function
def mytokenizer(x_train, y_train, x_val, y_val, max_len=200, vocab_size=2000):
    """Convert text to numerical sequences using a vocabulary built from the training data"""
    stop_words = set(stopwords.words('english'))
    
    # Process all training text to build vocabulary
    word_list = []
    for sent in x_train:
        preprocessed_sent = preprocess_string(sent)
        word_list.extend([word for word in preprocessed_sent.split() if word and word not in stop_words])

    # Create vocabulary from most frequent words
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:vocab_size]
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}
    
    # Convert training data to sequences
    final_list_train = []
    for sent in x_train:
        preprocessed_sent = preprocess_string(sent).split()
        final_list_train.append([onehot_dict.get(word, 0) for word in preprocessed_sent if word in onehot_dict])

    # Convert validation data to sequences
    final_list_test = []
    for sent in x_val:
        preprocessed_sent = preprocess_string(sent).split()
        final_list_test.append([onehot_dict.get(word, 0) for word in preprocessed_sent if word in onehot_dict])

    # Pad sequences to uniform length
    final_list_train = pad(final_list_train, max_len)
    final_list_test = pad(final_list_test, max_len)

    # Encode sentiment labels (positive=1, negative=0)
    encoded_train = np.array([1 if label == 'positive' else 0 for label in y_train])
    encoded_test = np.array([1 if label == 'positive' else 0 for label in y_val])

    return final_list_train, encoded_train, final_list_test, encoded_test, onehot_dict

# Process the data
max_seq_len = 200
X_train, y_train, X_test, y_test, vocab = mytokenizer(
    X_traindata, y_traindata, X_testdata, y_testdata, max_len=max_seq_len
)

# Display statistics
review_lengths = [len(i) for i in X_traindata]
print(f"Average Review Length: {np.mean(review_lengths):.2f}")
print(f"Maximum Review Length: {np.max(review_lengths)}")
print(f"Vocabulary Size: {len(vocab) + 1}")  # +1 for padding token

# Define the LSTM model for sentiment analysis
class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.3):
        super(SentimentAnalysisModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Save dimensions for hidden state initialization
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        # Embed the input
        embedded = self.embedding(x)
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout and pass through final layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).long()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).long()
y_test_tensor = torch.from_numpy(y_test).float()

# Create dataset and dataloader objects
train_data = TensorDataset(X_train_tensor, y_train_tensor)
valid_data = TensorDataset(X_test_tensor, y_test_tensor)

# Set hyperparameters
batch_size = 64
vocab_size = len(vocab) + 1  # Add 1 for padding token
embedding_dim = 100
hidden_dim = 256
output_dim = 1
n_layers = 2
dropout = 0.3
learning_rate = 0.001
epochs = 10
clip_value = 5

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize model
model = SentimentAnalysisModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    dropout=dropout
)
model.to(device)
print(model)

# Define accuracy function
def calculate_accuracy(predictions, labels):
    """Calculate the accuracy of predictions"""
    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).float()
    return correct.sum() / len(correct)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training and evaluation loop
best_valid_loss = float('inf')
train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

# Create directory for saving model
save_dir = 'data/models'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/imdb_sentiment_model.pt'

for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0
    epoch_train_acc = 0
    
    # Get initial hidden state
    h = model.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Detach hidden state from history
        h = tuple([each.detach() for each in h])
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output, h = model(inputs, h)
        
        # Calculate loss and accuracy
        loss = criterion(output.squeeze(), labels)
        acc = calculate_accuracy(output.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        epoch_train_loss += loss.item()
        epoch_train_acc += acc.item()
    
    # Compute average loss and accuracy for the epoch
    epoch_train_loss /= len(train_loader)
    epoch_train_acc /= len(train_loader)
    
    # Validation phase
    model.eval()
    epoch_valid_loss = 0
    epoch_valid_acc = 0
    
    # Get initial hidden state for validation
    h = model.init_hidden(batch_size)
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Detach hidden state from history
            h = tuple([each.detach() for each in h])
            
            # Forward pass
            output, h = model(inputs, h)
            
            # Calculate loss and accuracy
            loss = criterion(output.squeeze(), labels)
            acc = calculate_accuracy(output.squeeze(), labels)
            
            # Update metrics
            epoch_valid_loss += loss.item()
            epoch_valid_acc += acc.item()
    
    # Compute average loss and accuracy for the epoch
    epoch_valid_loss /= len(valid_loader)
    epoch_valid_acc /= len(valid_loader)
    
    # Store metrics
    train_losses.append(epoch_train_loss)
    valid_losses.append(epoch_valid_loss)
    train_accs.append(epoch_train_acc)
    valid_accs.append(epoch_valid_acc)
    
    # Print epoch statistics
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%')
    print(f'Valid Loss: {epoch_valid_loss:.4f} | Valid Acc: {epoch_valid_acc*100:.2f}%')
    
    # Save model if validation loss improves
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

print('\nTraining complete!')

# Load the best model
model.load_state_dict(torch.load(save_path, weights_only=True))

# Function to predict sentiment for a new review
def predict_sentiment(model, review_text, vocab, max_len=200):
    """Predict sentiment (positive/negative) for a new review"""
    model.eval()
    
    # Preprocess the review
    preprocessed = preprocess_string(review_text).split()
    
    # Convert to numerical sequence
    seq = [vocab.get(word, 0) for word in preprocessed if word in vocab]
    
    # Pad sequence
    padded = np.zeros(max_len, dtype=int)
    if len(seq) > 0:
        padded[-min(len(seq), max_len):] = seq[:max_len]
    
    # Convert to tensor
    tensor = torch.LongTensor(padded).unsqueeze(0).to(device)
    
    # Initialize hidden state
    h = model.init_hidden(1)
    
    # Get prediction
    with torch.no_grad():
        output, _ = model(tensor, h)
    
    # Return prediction and confidence
    prediction = torch.round(output.squeeze()).item()
    confidence = output.squeeze().item()
    
    return "positive" if prediction == 1 else "negative", confidence

# Example usage
sample_review = "This movie was fantastic! I really enjoyed the acting and the plot."
sentiment, confidence = predict_sentiment(model, sample_review, vocab, max_seq_len)
print(f"\nSample Review: '{sample_review}'")
print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")



import pickle

# After processing data with mytokenizer
vocab_path = 'data/models/imdb_vocab.pkl'
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print(f"Vocabulary saved to {vocab_path}")
