import pandas as pd
from utilities.preprocessing import preprocess_text, tokenize
from models.naive_bayes import NaiveBayesClassifier
from models.generator import NaiveBayesTextGenerator

def load_data(full_file, train_ratio=0.8):
    df = pd.read_csv(full_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = [tokenize(preprocess_text(text)) for text in train_df["review"]]
    y_train = train_df["sentiment"].tolist()

    X_test = [tokenize(preprocess_text(text)) for text in test_df["review"]]
    y_test = test_df["sentiment"].tolist()

    return X_train, y_train, X_test, y_test

def main():
    dataset_file = "data/IMDB Dataset.csv"
    X_train, y_train, X_test, y_test = load_data(dataset_file)

    # Train NB classifier
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train, y_train)

    # Evaluate
    acc = nb.score(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Generator
    generator = NaiveBayesTextGenerator(nb)
    generator._build_bigrams(X_train, y_train)

    print("\nGenerated Positive Review (Unigram):")
    print(generator.generate_unigram("positive", length=20))

    print("\nGenerated Negative Review (Unigram):")
    print(generator.generate_unigram("negative", length=20))

    print("\nGenerated Positive Review (Bigram):")
    print(generator.generate_bigram("positive", length=20))

    print("\nGenerated Negative Review (Bigram):")
    print(generator.generate_bigram("negative", length=20))

if __name__ == "__main__":
    main()
