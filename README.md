# Kindle Sentiment Analysis

## Overview

This project aims to perform sentiment analysis on Kindle book reviews using various Natural Language Processing (NLP) techniques and machine learning models. The goal is to classify reviews as positive or negative based on their content.

## Project Structure

- **Data Preprocessing**: The dataset consists of thousands of Kindle book reviews. The raw text data was cleaned and preprocessed by removing special characters, stopwords, URLs, HTML tags, and performing lemmatization.
  
- **Feature Extraction**: 
  - **Bag of Words (BoW)**: Converted the preprocessed text into a sparse matrix of token counts.
  - **TF-IDF**: Generated term frequency-inverse document frequency features to reflect the importance of words.
  - **Word2Vec**: Used the Word2Vec model to create word embeddings, capturing the semantic context of the text.

- **Modeling**:
  - **Random Forest Classifier**: Trained on each set of features (BoW, TF-IDF, Word2Vec) to predict the sentiment of reviews.
  - **Evaluation**: Achieved accuracy rates up to 80% with the BoW model, with comprehensive evaluation metrics including precision, recall, and F1-score.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Libraries: `numpy`, `pandas`, `matplotlib`, `nltk`, `sklearn`, `gensim`, `bs4`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Asaha9345/kindle_sentiment_analysis.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Kindle_Sentiment_analysis.ipynb
   ```

### Dataset

The dataset is included in the project and consists of Kindle book reviews with ratings. You can find the dataset file [here](link-to-dataset-if-available).

## Usage

- **Exploratory Data Analysis (EDA)**: Visualize and understand the distribution of sentiments within the dataset.
- **Feature Engineering**: Explore different methods of converting text into numerical data.
- **Model Training**: Experiment with various models and parameters to improve prediction accuracy.
- **Model Evaluation**: Use confusion matrices, classification reports, and accuracy metrics to assess model performance.

## Results

- **Bag of Words**: Achieved an accuracy of 80%, with high precision for positive sentiments.
- **TF-IDF**: Provided a balanced performance with an accuracy of 79%.
- **Word2Vec**: Showed an accuracy of 77%, with better recall for negative sentiments.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- **NLTK** for providing powerful text processing tools.
- **scikit-learn** for its comprehensive machine learning algorithms.
- **gensim** for its robust implementation of Word2Vec.
- **Google Colab** for providing an accessible environment to run the project.
- **Krish Naik** for the amezing explanation about the things.
