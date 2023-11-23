from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Naive Bayes Classifier with given smoothing parameter.

        Parameters:
            alpha : Smoothing parameter for Laplace smoothing.
        """

        self.alpha = alpha
        self.classes_ = None
        self._class_probabilities = None
        self._priors = None

    def fit(self, X, y) -> object:
        """
        Fit the Naive Bayes Classifier on given training data.

        Parameters:
            X : Feature matrix for training instances.
            y : Class labels for training instances.

        Returns:
            self : Fitted classifier instance.
        """

        self.classes_, label_counts = np.unique(y, return_counts=True)
        self.classes_ = np.array(sorted(self.classes_))
        self._class_probabilities = np.zeros((len(self.classes_), X.shape[1]))
        self._priors = np.zeros(len(self.classes_))
        for i, (label, label_count) in enumerate(zip(self.classes_, label_counts)):
            data_for_label = X[y == label]
            feature_counts = data_for_label.sum(axis=0)
            feature_counts_smoothed = feature_counts + self.alpha
            probabilities = np.log(feature_counts_smoothed) - np.log(np.sum(feature_counts_smoothed))
            self._class_probabilities[i] = probabilities
            prior = np.log(label_count) - np.log(len(y))
            self._priors[i] = prior

        return self

    def predict(self, X) -> np.array:
        """
        Predict class labels for given instances.

        Parameters:
            X : Feature matrix for instances to be predicted.

        Returns:
            predicted_class_labels : Predicted class labels for the instances.
        """

        log_prob_matrix = X.dot(self._class_probabilities.T)
        log_prob_matrix += self._priors
        log_prob_matrix = np.argmax(log_prob_matrix, axis=1)
        predicted_class_labels = self.classes_[log_prob_matrix]
        return predicted_class_labels

    def predict_proba(self, X) -> np.array:
        """
        Predict class probabilities for given instances.

        Parameters :
            X : Feature matrix for instances to be predicted.

        Returns :
            normalized_prob_matrix: Class probability matrix for the instances.
        """

        log_prob_matrix = X.dot(self._class_probabilities.T)
        log_prob_matrix += self._priors  # same as predict so far
        log_prob_matrix -= np.max(log_prob_matrix, axis=1, keepdims=True)  # we exponentiate soon, this makes it easier
        prob_matrix = np.exp(log_prob_matrix)  # turn back to original probabilities
        row_sums = np.sum(prob_matrix, axis=1, keepdims=True)  # normalize by sum
        normalized_prob_matrix = prob_matrix / row_sums
        return normalized_prob_matrix
