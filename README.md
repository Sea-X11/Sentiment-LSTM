# Sentiment Analysis FastAPI Project

A powerful Sentiment Analysis API built with **FastAPI** and **PyTorch**, utilizing a pre-trained LSTM model to predict the sentiment (positive or negative) of text input. This project provides both a web interface and a RESTful API for sentiment prediction, along with an optional CLI mode for local testing.

---

## Features

- **RESTful API**: Predict sentiment via a simple GET endpoint (`/predict`).
- **Web Interface**: Interactive HTML frontend to input text and visualize results with confidence scores.
- **CLI Mode**: Command-line interface for quick local testing.
- **LSTM Model**: Pre-trained PyTorch model for accurate sentiment classification.
- **Confidence Scoring**: Returns sentiment predictions with confidence percentages.
- **Responsive Design**: Clean and modern UI with a confidence meter visualization.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9.0+ (with or without CUDA support)
- FastAPI and Uvicorn
- Git (optional, for cloning the repo)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sea-X11/Sentiment-LSTM.git
   cd Sentiment-LSTM
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   *Note*: Ensure you have a `requirements.txt` file. Here's a sample:
   ```
   fastapi==0.68.0
   uvicorn==0.15.0
   torch==1.9.0
   pydantic==1.8.2
   jinja2==3.0.1
   ```

4. **Download Pre-trained Model and Vocabulary**:
   - Place `imdb_sentiment_model.pt` and `imdb_vocab.pkl` in the root directory (or update `MODEL_CONFIG` paths in the code if stored elsewhere).
   - These files are required for the LSTM model and tokenization.

5. **Run the Application**:
   - **API Mode** (default):
     ```bash
     uvicorn main:app --host 0.0.0.0 --port 8000 --reload
     ```
   - **CLI Mode**:
     ```bash
     python main.py --cli
     ```

---

## Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8000/`.
2. Enter text in the provided textarea.
3. Click "Analyze Sentiment" to see the result, including sentiment (POSITIVE/NEGATIVE) and confidence score.

### API Endpoint
- **Endpoint**: `GET /predict`
- **Parameters**: `text` (string)
- **Example**:
  ```bash
  curl "http://localhost:8000/predict?text=I%20love%20this%20movie!"
  ```
- **Response**:
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.95
  }
  ```

### CLI Mode
- Run `python main.py --cli`.
- Enter text when prompted, and receive sentiment and confidence output. Type `q` to quit.

---

## Project Structure

```
sentiment-analysis-api/
├── main.py              # Main application file
├── model.py             # Model definition and prediction logic (assumed)
├── static/              # Static files (CSS, etc.)
│   └── css/
│       └── styles.css
├── templates/           # HTML templates 
│   └── index.html       # Can be automatically generated
├── imdb_sentiment_model.pt  # Pre-trained model 
├── imdb_vocab.pkl       # Vocabulary file 
└── README.md            # This file
└── render.yaml          # Deploy on render
├── requirements.txt        # Dependencies
├── sentiment-inference-run locally.py                 #  easy run locally  (optional)
├── trainning-model  (you can train by yourself).py    #  train by yoursel  (optional)

```

## Configuration

The model configuration is defined in `main.py` under `MODEL_CONFIG`:
- `vocab_size`: 2001 (includes padding token)
- `embedding_dim`: 100
- `hidden_dim`: 256
- `output_dim`: 1
- `n_layers`: 2
- `dropout`: 0.3
- `max_seq_len`: 200
- `model_path`: "imdb_sentiment_model.pt"
- `vocab_path`: "imdb_vocab.pkl"

Adjust these values or paths as needed for your model.

---

## Development

- **Dependencies**: Managed via FastAPI's `Depends` for model and vocabulary loading.
- **Templates**: Uses Jinja2 for rendering the web interface.
- **Static Files**: CSS is served from the `static/` directory.
- **Startup**: Model and vocabulary are loaded on app startup (`@app.on_event("startup")`).

To extend the project:
1. Modify `model.py` (assumed) to update the LSTM architecture or prediction logic.
2. Enhance the UI in `templates/index.html`.
3. Add more API endpoints in `main.py`.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)


