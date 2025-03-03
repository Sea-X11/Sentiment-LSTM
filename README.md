# flask-sentiment-LSTM
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
