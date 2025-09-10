import numpy as np
from collections import defaultdict, Counter
import random

class NaiveBayesTextGenerator:
    def __init__(self, nb_model):
        self.nb_model = nb_model
        self.bigram_counts = {}
        self.unigram_counts = {}

    def _build_bigrams(self, X, y):
        """
        Build bigram counts for each class.
        """
        classes = set(y)
        self.bigram_counts = {c: defaultdict(Counter) for c in classes}
        self.unigram_counts = {c: Counter() for c in classes}

        for doc, label in zip(X, y):
            for i in range(len(doc) - 1):
                w1, w2 = doc[i], doc[i+1]
                self.bigram_counts[label][w1][w2] += 1
                self.unigram_counts[label][w1] += 1

    def generate_unigram(self, target_class, length=20, temperature=1.0):
        """
        Generate text using unigram distribution (baseline).
        """
        probs = np.array(list(self.nb_model.word_probs[target_class].values()))
        vocab = list(self.nb_model.word_probs[target_class].keys())

        if temperature != 1.0:
            probs = np.log(probs + 1e-12) / temperature
            probs = np.exp(probs)
            probs /= probs.sum()

        words = np.random.choice(vocab, size=length, replace=True, p=probs)
        return " ".join(words)

    def generate_bigram(self, target_class, length=20, alpha=1.0):
        """
        Generate text using bigram distribution with Laplace smoothing.
        """
        vocab = list(self.nb_model.vocab)
        w1 = random.choice(vocab)
        words = [w1]

        for _ in range(length - 1):
            candidates = self.bigram_counts[target_class][w1]
            total = self.unigram_counts[target_class][w1]

            if total == 0:
                # If no continuation, pick random word from vocab
                w2 = random.choice(vocab)
            else:
                # Build smoothed distribution over observed bigrams
                vocab_list = list(candidates.keys())
                counts = np.array([candidates[w] for w in vocab_list], dtype=float)
                probs = (counts + alpha) / (total + alpha * len(vocab_list))
                probs /= probs.sum()

                w2 = np.random.choice(vocab_list, p=probs)

            words.append(w2)
            w1 = w2

        return " ".join(words)
