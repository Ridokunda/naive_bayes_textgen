import numpy as np
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        """
        Multinomial Naive Bayes implementation.
        :param alpha: Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.class_priors = {}      # P(c)
        self.word_probs = {}        # P(w|c)
        self.vocab = set()
        self.class_word_counts = {} # total word counts per class
        self.vocab_size = 0

    def fit(self, X, y):
        """
        Train Naive Bayes model.
        :param X: list of documents (each doc is list of words)
        :param y: list of labels (same length as X)
        """
        # Count docs per class
        class_counts = Counter(y)
        total_docs = len(y)
        self.class_priors = {c: class_counts[c] / total_docs for c in class_counts}

        # Count words per class
        word_counts = {c: defaultdict(int) for c in class_counts}
        class_word_totals = {c: 0 for c in class_counts}

        for doc, label in zip(X, y):
            for word in doc:
                word_counts[label][word] += 1
                class_word_totals[label] += 1
                self.vocab.add(word)

        self.vocab_size = len(self.vocab)
        self.class_word_counts = class_word_totals

        # Compute smoothed probabilities P(w|c)
        self.word_probs = {
            c: {
                w: (word_counts[c][w] + self.alpha) /
                   (class_word_totals[c] + self.alpha * self.vocab_size)
                for w in self.vocab
            }
            for c in class_counts
        }

    def predict(self, doc):
        """
        Predict class for a single document.
        :param doc: list of words
        """
        scores = {}
        for c in self.class_priors:
            # log P(c)
            log_prob = np.log(self.class_priors[c])

            # add log P(w|c) for each word
            for word in doc:
                if word in self.vocab:
                    log_prob += np.log(self.word_probs[c].get(word, self.alpha / 
                                    (self.class_word_counts[c] + self.alpha * self.vocab_size)))
            scores[c] = log_prob

        return max(scores, key=scores.get)

    def score(self, X, y):
        """
        Compute accuracy on test set.
        """
        correct = 0
        for doc, label in zip(X, y):
            if self.predict(doc) == label:
                correct += 1
        return correct / len(y)
