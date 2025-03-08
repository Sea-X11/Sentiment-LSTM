import torch
import numpy as np
import re
import os

# Define the same preprocessing function used during training
def preprocess_string(s):
    """Clean and normalize text by removing non-alphabetic characters and converting to lowercase"""
    s = re.sub(r"[^a-zA-Z\s]", '', s)  # Keep letters and spaces
    s = re.sub(r"\s+", ' ', s)  # Normalize whitespaceh
    return s.strip().lower()  # Trim and convert to lowercase

# Define the same model architecture used during training
class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.3):
        super(SentimentAnalysisModel, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        
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
        device = next(self.parameters()).device
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# Function to load the vocabulary from a file
def load_vocabulary(vocab_path):
    """Load vocabulary dictionary from a file"""
    import pickle
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)

# Function to save vocabulary to a file
def save_vocabulary(vocab, vocab_path):
    """Save vocabulary dictionary to a file"""
    import pickle
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

# Inference function
def predict_sentiment(model, text, vocab, max_len=200):
    """
    Predict sentiment for a given text using a trained model
    
    Args:
        model: Trained PyTorch model
        text: Text to analyze
        vocab: Dictionary mapping words to indices
        max_len: Maximum sequence length
        
    Returns:
        sentiment: "positive" or "negative"
        confidence: Confidence score (0-1)
    """
    model.eval()  # Set model to evaluation mode
    
    # Get device
    device = next(model.parameters()).device
    
    # Preprocess the text
    preprocessed = preprocess_string(text).split()
    
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

# Main inference script
def main():
    # Set parameters - make sure these match your training parameters
    vocab_size = 2001  # Vocabulary size + 1 for padding
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    dropout = 0.3
    max_seq_len = 200
    
    # Model and vocabulary paths
    model_path = 'imdb_sentiment_model.pt'
    vocab_path = 'imdb_vocab.pkl'
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    model = SentimentAnalysisModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded from {model_path}")
    
    # Load vocabulary
    try:
        vocab = load_vocabulary(vocab_path)
        print(f"Vocabulary loaded from {vocab_path}")
    except FileNotFoundError:
        print(f"Vocabulary file not found at {vocab_path}")
        print("Make sure to save your vocabulary during training using save_vocabulary() function")
        return
    
    # Interactive mode
    print("\n=== IMDB Sentiment Analysis Model ===")
    print("Enter text for sentiment analysis (or 'q' to quit):")
    
    while True:
        user_input = input("\nEnter review text: ")
        
        if user_input.lower() == 'q':
            break
        
        if not user_input.strip():
            print("Please enter some text to analyze.")
            continue
        
        # Predict sentiment
        sentiment, confidence = predict_sentiment(model, user_input, vocab, max_seq_len)
        
        # Display results
        print(f"Sentiment: {sentiment.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        
        # Provide more context based on confidence
        if confidence > 0.9:
            strength = "Very strong"
        elif confidence > 0.75:
            strength = "Strong"
        elif confidence > 0.6:
            strength = "Moderate"
        elif confidence > 0.5:
            strength = "Weak"
        else:
            strength = "Very weak"
        
        print(f"Strength of sentiment: {strength}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
