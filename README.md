# sentence-sentiment: Deep Learning Sentiment Classifier

## Overview

This project implements a deep learning-based sentiment analysis model using Python to classify text into six emotional categories: **Sadness**, **Joy**, **Love**, **Anger**, **Fear**, and **Surprise**. Built with TensorFlow and Keras, it includes comprehensive data preprocessing, model training, evaluation, and a prediction function for real-time sentiment analysis. The project leverages libraries such as Pandas, NumPy, Matplotlib, Seaborn, NLTK, and Scikit-learn for data handling, visualization, and NLP tasks.

## Features

- **Data Preprocessing**: Cleans text by removing URLs, special characters, numbers, and stopwords, and converts it to lowercase.
- **Exploratory Data Analysis**: Visualizes sentiment distribution with bar plots and pie charts, addressing dataset imbalance with class weights.
- **Model Architecture**: A Sequential model with Embedding, Bidirectional LSTM, BatchNormalization, Dropout, and Dense layers.
- **Training**: Trains the model for 5 epochs with class weights, using an 80-20 train-test split.
- **Evaluation**: Assesses performance with accuracy, loss, confusion matrix, and classification report.
- **Visualization**: Generates plots for sentiment distribution, training/validation accuracy/loss, and confusion matrix.
- **Prediction**: Provides a `predict` function to classify new text inputs with probability visualizations.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/john-thomas-ml/sentence-sentiment.git
   ```

2. **Install Dependencies**: Install the required Python libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow nltk scikit-learn
   ```

3. **Download NLTK Data**: Download the stopwords dataset:

   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Prepare the Dataset**: Ensure `text.csv` is in the project directory, containing columns `text` (input text) and `label` (sentiment labels 0-5).

## Usage

1. **Run the Notebook**: Open and execute the `sentiment.ipynb` Jupyter Notebook to preprocess data, train the model, and generate visualizations:

   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook

2. **Predict Sentiment**: Use the `predict` function to classify new text:

   ```python
   text = "I am very happy!"
   predict(text)
   ```

   This outputs a bar chart of sentiment probabilities.

3. **Load Saved Model**: The trained model is saved as `sentiment.h5`. Load it for predictions:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model("sentiment.h5")
   ```

## Dataset

- **Format**: CSV file (`text.csv`) with columns:
  - `text`: Raw text input.
  - `label`: Sentiment labels (0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise).
- **Preprocessing**:
  - Drops `Unnamed: 0` column if present.
  - Removes URLs, special characters, numbers, and stopwords.
  - Converts text to lowercase.
  - Tokenizes and pads sequences for model input.
- **Distribution**: Unbalanced (e.g., Joy: 33.8%, Sadness: 29.1%), handled with class weights.

## Model Architecture

The model is a Sequential neural network with:

1. **Embedding**  
   `Embedding(input_size, 100)`  
   Converts input tokens into dense 100-dimensional vectors.

2. **Bidirectional LSTM (Layer 1)**  
   `Bidirectional(LSTM(128, return_sequences=True))`  
   Processes the input sequence in both forward and backward directions, returning the full sequence.

3. **Batch Normalization**  
   `BatchNormalization()`  
   Normalizes activations to stabilize and speed up training.

4. **ReLU Activation**  
   `ReLU()`  
   Applies the ReLU activation function to introduce non-linearity.

5. **Dropout (20%)**  
   `Dropout(0.2)`  
   Randomly deactivates 20% of neurons to help prevent overfitting.

6. **Bidirectional LSTM (Layer 2)**  
   `Bidirectional(LSTM(64))`  
   Further processes the features in both directions, outputting the final hidden state.

7. **Dropout (30%)**  
   `Dropout(0.3)`  
   Adds additional regularization by randomly dropping 30% of neurons.

8. **Output Layer**  
   `Dense(6, activation='softmax')`  
   Produces probability distributions over 6 output classes.

### Training Details

- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy
- **Metrics**: Accuracy
- **Epochs**: 5
- **Batch Size**: 128
- **Class Weights**: Inverse frequency-based (e.g., Joy: 1/0.338, Sadness: 1/0.291).

## Results

- **Performance**: Achieves \~94-95% validation accuracy after 5 epochs.
- **Evaluation**: Includes confusion matrix and classification report for detailed metrics.
- **Visualizations**:
  - Bar plot and pie chart of sentiment distribution.
  - Training/validation accuracy and loss plots.
  - Confusion matrix heatmap.
