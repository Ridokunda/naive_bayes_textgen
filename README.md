# Naive Bayes Text Generation

A simple Python project for sentiment classification and text generation using Naive Bayes and bigram models, trained on the IMDB movie review dataset.

## Features
- **Text Preprocessing**: Cleans and tokenizes raw text for modeling.
- **Naive Bayes Classifier**: Multinomial Naive Bayes for sentiment classification (positive/negative).
- **Text Generation**: Generates synthetic reviews using unigram and bigram models, conditioned on sentiment.

## Project Structure
```
main.py                  # Main script: trains, evaluates, and generates text
models/
    naive_bayes.py       # Naive Bayes classifier implementation
    generator.py         # Text generator using unigram/bigram models
utilities/
    preprocessing.py     # Text cleaning and tokenization functions
data/
    IMDB Dataset.csv     # Movie review dataset (CSV)
```

## Requirements
- Python 3.7+
- pandas
- numpy

Install dependencies:
```powershell
pip install pandas numpy
```

## Usage
1. Place the IMDB dataset CSV file in the `data/` folder (filename: `IMDB Dataset.csv`).
2. Run the main script:
```powershell
python main.py
```

## Output
- Prints test accuracy of the classifier.
- Generates sample positive and negative reviews using both unigram and bigram models.

## Customization
- Adjust the Laplace smoothing parameter (`alpha`) in `main.py` for the classifier.
- Change the length or temperature of generated text in the generator calls.

## File Descriptions
- **main.py**: Loads data, trains classifier, evaluates, and generates text.
- **models/naive_bayes.py**: Implements the Naive Bayes classifier.
- **models/generator.py**: Implements text generation using unigram and bigram models.
- **utilities/preprocessing.py**: Functions for cleaning and tokenizing text.


