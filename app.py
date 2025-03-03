import os
import pickle
import torch
import numpy as np
import re
from flask import Flask, request, jsonify, render_template

# Define the same preprocessing function
def preprocess_string(s):
    """Clean and normalize text by removing non-alphabetic characters and converting to lowercase"""
    s = re.sub(r"[^a-zA-Z\s]", '', s)  # Keep letters and spaces
    s = re.sub(r"\s+", ' ', s)  # Normalize whitespace
    return s.strip().lower()  # Trim and convert to lowercase

# Define the model architecture
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

# Prediction function
def predict_sentiment(model, text, vocab, max_len=200):
    """Predict sentiment for a given text"""
    model.eval()
    
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
    confidence = float(output.squeeze().item())  # Convert to Python float for JSON serialization
    
    sentiment = "positive" if prediction == 1 else "negative"
    
    # Determine sentiment strength
    if confidence > 0.9 or confidence < 0.1:
        strength = "Very strong"
    elif confidence > 0.75 or confidence < 0.25:
        strength = "Strong"
    elif confidence > 0.6 or confidence < 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    # Adjust confidence to be relative to prediction
    display_confidence = confidence if sentiment == "positive" else 1 - confidence
    
    return {
        "sentiment": sentiment,
        "confidence": display_confidence,
        "confidence_percent": f"{display_confidence * 100:.1f}%",
        "strength": strength
    }

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and vocabulary
model = None
vocab = None

# Load model and vocabulary
def load_model():
    global model, vocab
    
    # Set parameters - make sure these match your training parameters
    vocab_size = 2001  # Vocabulary size + 1 for padding
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    dropout = 0.3
    
    # Model and vocabulary paths (adjust as needed)
    model_path = os.environ.get('MODEL_PATH', 'data/models/imdb_sentiment_model.pt')
    vocab_path = os.environ.get('VOCAB_PATH', 'data/models/imdb_vocab.pkl')
    
    # Check if GPU is available (but use CPU for deployment simplicity)
    device = torch.device('cpu')  # For deployment, CPU is often more reliable
    
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
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {vocab_path}")
    except FileNotFoundError:
        print(f"Vocabulary file not found at {vocab_path}")
        raise

# API route for sentiment analysis
@app.route('/api/analyze', methods=['POST'])
def analyze():
    global model, vocab
    
    # Load model if not already loaded
    if model is None or vocab is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Check if request contains JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Extract text from request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text']
    
    # Perform sentiment analysis
    try:
        result = predict_sentiment(model, text, vocab)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route for web interface
@app.route('/')
def home():
    # Check if the template file exists
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        # Create the template first
        create_template()
        
    return render_template('index.html')

# Create a simple HTML template
@app.route('/create_template')
def create_template():
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Write the template file
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .positive {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
        }
        .negative {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis</h1>
        <p>Enter some text to analyze its sentiment:</p>
        
        <textarea id="text-input" placeholder="Type or paste your text here..."></textarea>
        <button onclick="analyzeSentiment()">Analyze</button>
        
        <div id="loader" class="loader"></div>
        
        <div id="result">
            <h3>Analysis Result:</h3>
            <p><strong>Sentiment:</strong> <span id="sentiment"></span></p>
            <p><strong>Confidence:</strong> <span id="confidence"></span></p>
            <p><strong>Strength:</strong> <span id="strength"></span></p>
        </div>
    </div>

    <script>
        function analyzeSentiment() {
            const text = document.getElementById('text-input').value.trim();
            
            if (text === '') {
                alert('Please enter some text to analyze');
                return;
            }
            
            // Show loader
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                // Show result
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = data.sentiment;
                
                document.getElementById('sentiment').textContent = data.sentiment.toUpperCase();
                document.getElementById('confidence').textContent = data.confidence_percent;
                document.getElementById('strength').textContent = data.strength;
            })
            .catch(error => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred during analysis');
            });
        }
    </script>
</body>
</html>
        ''')
    
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    # Check if running in production (e.g., on Render)
    if os.environ.get('RENDER', False):
        # In production, let the platform handle the serving
        # But load the model when the app starts
        with app.app_context():
            try:
                load_model()
            except Exception as e:
                print(f"Error loading model: {e}")
    else:
        # For local development
        # Load model lazily on first request instead of at startup
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))