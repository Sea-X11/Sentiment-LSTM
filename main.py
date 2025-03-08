from typing import Union, Dict
import os
import torch
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from model import SentimentAnalysisModel, predict_sentiment, load_vocabulary

# Configuration
MODEL_CONFIG = {
    "vocab_size": 2001,  # Vocabulary size + 1 (padding)
    "embedding_dim": 100,
    "hidden_dim": 256,
    "output_dim": 1,
    "n_layers": 2,
    "dropout": 0.3,
    "max_seq_len": 200,
    "model_path": "imdb_sentiment_model.pt",
    "vocab_path": "imdb_vocab.pkl"
}

# Response model
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using a trained LSTM model",
    version="1.0.0"
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Set up static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    # Create directories if they don't exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Create the template file
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 100px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result-container {
            display: {% if result %}block{% else %}none{% endif %};
            background-color: #f0f7ff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .positive {
            color: green;
            font-weight: bold;
        }
        .negative {
            color: red;
            font-weight: bold;
        }
        .confidence-meter {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 4px;
            margin-top: 10px;
        }
        .confidence-bar {
            height: 20px;
            background-color: {% if result and result.sentiment == "positive" %}#4CAF50{% else %}#f44336{% endif %};
            border-radius: 4px;
            width: {{ confidence_percentage }}%;
            text-align: center;
            line-height: 20px;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Sentiment Analysis Tool</h1>

    <div class="project-info">
        <p>This is a Sentiment Analysis Model built using PyTorch. It analyzes text to determine if it's positive or negative, focusing on Natural Language Processing (NLP). The model is trained on the IMDB Dataset of 50K Movie Reviews and utilizes an LSTM neural network. You can find the project's source code on <a href="https://github.com/Sea-X11/Sentiment-LSTM" target="_blank">GitHub</a>.</p>
    </div>

    <div class="form-container">
        <form method="post">
            <p>Enter text to analyze sentiment:</p>
            <textarea name="text" placeholder="Type your text here...">{{ text }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>
    </div>

    <div class="result-container" id="result">
        {% if result %}
            <h2>Analysis Result:</h2>
            <p>Sentiment: <span class="{% if result.sentiment == 'positive' %}positive{% else %}negative{% endif %}">
                {{ result.sentiment|upper }}
            </span></p>
            <p>Confidence: {{ "%.1f"|format(result.confidence * 100) }}%</p>
            <div class="confidence-meter">
                <div class="confidence-bar">{{ "%.1f"|format(result.confidence * 100) }}%</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
    """)

# Create CSS file
os.makedirs("static/css", exist_ok=True)
with open("static/css/styles.css", "w") as f:
    f.write("""
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
    """)

# Dependency to get the model and vocabulary
def get_model_resources():
    """Dependency that provides the model and vocabulary."""
    return app.state.model, app.state.vocab, app.state.max_seq_len

@app.on_event("startup")
async def startup_event():
    """Initialize model and vocabulary when the API starts."""
    # Check device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SentimentAnalysisModel(
        vocab_size=MODEL_CONFIG["vocab_size"],
        embedding_dim=MODEL_CONFIG["embedding_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        output_dim=MODEL_CONFIG["output_dim"],
        n_layers=MODEL_CONFIG["n_layers"],
        dropout=MODEL_CONFIG["dropout"]
    )

    # Load model weights
    try:
        model.load_state_dict(torch.load(
            MODEL_CONFIG["model_path"], 
            map_location=device, 
            weights_only=True
        ))
        model.to(device)
        model.eval()  # Set model to evaluation mode
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {MODEL_CONFIG['model_path']}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    # Load vocabulary
    try:
        vocab = load_vocabulary(MODEL_CONFIG["vocab_path"])
    except FileNotFoundError:
        raise RuntimeError(f"Vocabulary file not found at {MODEL_CONFIG['vocab_path']}")
    except Exception as e:
        raise RuntimeError(f"Failed to load vocabulary: {str(e)}")
    
    # Store resources in app state
    app.state.model = model
    app.state.vocab = vocab
    app.state.max_seq_len = MODEL_CONFIG["max_seq_len"]
    app.state.device = device

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

@app.get("/predict", response_model=SentimentResponse)
def predict(text: str, resources=Depends(get_model_resources)):
    """
    Predict sentiment for the provided text.
    
    Args:
        text: The text to analyze
        
    Returns:
        A SentimentResponse with the sentiment and confidence
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    model, vocab, max_seq_len = resources
    
    try:
        sentiment, confidence = predict_sentiment(model, text, vocab, max_seq_len)
        return SentimentResponse(
            sentiment=sentiment,
            confidence=float(confidence)  # Ensure it's a Python float
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": "",
        "result": None,
        "confidence_percentage": 0
    })

@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, text: str = Form(...), resources=Depends(get_model_resources)):
    """Process the form submission and render the results"""
    if not text.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "result": None,
            "confidence_percentage": 0
        })
    
    model, vocab, max_seq_len = resources
    
    try:
        sentiment, confidence = predict_sentiment(model, text, vocab, max_seq_len)
        result = {
            "sentiment": sentiment,
            "confidence": float(confidence)
        }
        confidence_percentage = float(confidence) * 100
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "result": result,
            "confidence_percentage": confidence_percentage
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "result": {"sentiment": "error", "confidence": 0},
            "confidence_percentage": 0,
            "error": str(e)
        })

def start_cli_mode():
    """Start interactive CLI mode for sentiment analysis."""
    # Check if resources exist
    if not os.path.exists(MODEL_CONFIG["model_path"]):
        print(f"Model file not found at {MODEL_CONFIG['model_path']}")
        return
    if not os.path.exists(MODEL_CONFIG["vocab_path"]):
        print(f"Vocabulary file not found at {MODEL_CONFIG['vocab_path']}")
        return
    
    # Initialize resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SentimentAnalysisModel(
        vocab_size=MODEL_CONFIG["vocab_size"],
        embedding_dim=MODEL_CONFIG["embedding_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        output_dim=MODEL_CONFIG["output_dim"],
        n_layers=MODEL_CONFIG["n_layers"],
        dropout=MODEL_CONFIG["dropout"]
    )
    
    model.load_state_dict(torch.load(
        MODEL_CONFIG["model_path"], 
        map_location=device, 
        weights_only=True
    ))
    model.to(device)
    model.eval()
    
    vocab = load_vocabulary(MODEL_CONFIG["vocab_path"])
    max_seq_len = MODEL_CONFIG["max_seq_len"]
    
    # Interactive mode
    print("\n=== Sentiment Analysis Demo ===")
    print("Enter text to analyze sentiment (or 'q' to quit):")

    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'q':
            break
        if not user_input.strip():
            print("Please enter some text.")
            continue

        sentiment, confidence = predict_sentiment(model, user_input, vocab, max_seq_len)

        print(f"Sentiment: {sentiment.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode
        start_cli_mode()
    else:
        # API mode
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
