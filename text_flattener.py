import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TextFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array(X).flatten().astype(str).tolist()
