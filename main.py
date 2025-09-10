import pandas as pd
from utils.preprocessing import preprocess_text, tokenize
from models.naive_bayes import NaiveBayesClassifier

def load_data(train_file, test_file):
    """
    Loads train and test datasets.
    Assumes CSV with columns: [review, label]
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = [tokenize(preprocess_text(text)) for text in train_df["review"]]
    y_train = train_df["label"].tolist()

    X_test = [tokenize(preprocess_text(text)) for text in test_df["review"]]
    y_test = test_df["label"].tolist()

    return X_train, y_train, X_test, y_test

def main():
    # Load dataset
    X_train, y_train, X_test, y_test = load_data("data/imdb_train.csv", "data/imdb_test.csv")

    # Train NB classifier
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train, y_train)

    # Evaluate
    acc = nb.score(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Try classifying a new sentence
    sample = "I really loved this movie, it was amazing!"
    processed = tokenize(preprocess_text(sample))
    prediction = nb.predict(processed)
    print(f"Sample: {sample}")
    print(f"Predicted Sentiment: {prediction}")

if __name__ == "__main__":
    main()
