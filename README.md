# ssentiment-LSTM
The model in Flask application is a Sentiment Analysis Model built using PyTorch. 
        <h3>Model Overview:</h3>
        <p>This application is Sentiment Analysis Model built with <strong>PyTorch</strong>. The model is part of the Artificial Intelligence <strong>(AI)</strong> and <strong>Machine Learning</strong>(ML) fields, specifically focusing on Natural Language Processing (NLP). It analyzes text to determine its sentiment as positive or negative.</p>
        <p><strong>Data Source:</strong> The model is trained using the <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews" target="_blank">IMDB Dataset of 50K Movie Reviews</a>, which contains movie reviews used for sentiment analysis. This dataset is commonly used for sentiment analysis in the movie domain.</p>
        <h3>Model Architecture:</h3>
        <ul>
            <li>The model is a Long Short-Term Memory (LSTM) neural network.</li>
            <li>It uses an <strong>embedding layer</strong> to transform words into vector representations.</li>
            <li>The <strong>LSTM</strong> layer processes the sequential input (the word embeddings).</li>
            <li>The output from the last time step is passed through a <strong>dropout</strong> and a <strong>fully connected layer</strong>.</li>
        </ul>
        <h3>Model Prediction Function:</h3>
        <ul>
            <li>Text is preprocessed, tokenized, and converted into numerical sequences based on a vocabulary.</li>
            <li>The sequence is padded and passed through the model to predict sentiment.</li>
            <li>The result includes the <strong>sentiment</strong> (positive or negative), the <strong>confidence</strong> score, and the <strong>strength</strong> of the prediction (strong, moderate, weak).</li>
        </ul>

# Deployment Guide:Setting Up Your Flask Application Locally

This guide walks you through setting up and deploying your sentiment analysis Flask application to cloud platforms.

### Step 1: Organize Your Project Structure
```
sentiment-app/
├── app.py                  # Main Flask application
├── requirements.txt        # Dependencies
├── data/
│   └── models/
│       ├── imdb_sentiment_model.pt  # Your trained model
│       └── imdb_vocab.pkl           # Your vocabulary
├── sentiment-inference-run locally.py                 #  (optional)
├── trainning-model(you can train by yourself).py                 #  (optional)
└── templates/
    └── index.html          # Web interface
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: run Locally
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser to test the application.

