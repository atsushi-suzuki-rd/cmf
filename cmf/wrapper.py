import numpy as np
from sklearn import linear_model
import pyper


class TimeSeriesLinearRegression(object):
    def __init__(self, window_width=5, **kwargs):
        self.window_width=window_width
        self.linear_regressor = linear_model.LinearRegression(**kwargs)

    def fit(self, X, Y, **kwargs):
        XX = self._sliding_duplicate(X)
        self.linear_regressor.fit(XX, Y)

    def predict(self, X, **kwargs):
        XX = self._sliding_duplicate(X)
        return self.linear_regressor.predict(XX)

    def _sliding_duplicate(self, X):
        return np.concatenate(
            [np.pad(np.array(X), pad_width=((self.window_width, self.window_width), (0, 0)), mode='constant', constant_values=0.)[self.window_width+i:-(self.window_width-i), :]
            for i in range(-self.window_width, self.window_width)],
            axis=1)


class MatrixCompletionRegression(object):
    def __init__(self, matrix_factorizer):
        self.matrix_factorizer = matrix_factorizer
        self.X_n_features = None
        self.Y_n_features = None

    def fit(self, X, Y, **kwargs):
        self.X_n_features = X.shape[1]
        self.Y_n_features = Y.shape[1]
        XY = np.concatenate([X, Y], axis=1)
        self.matrix_factorizer.fit(XY, **kwargs)

    def predict(self, X, **kwargs):
        n_samples = X.shape[0]
        XO = np.concatenate([X, np.zeros((n_samples, self.Y_n_features))], axis=1)
        F = np.concatenate([np.ones(X.shape), np.zeros((n_samples, self.Y_n_features))], axis=1)
        Z = self.matrix_factorizer.transform(XO, filtre=F)
        XY_hat = self.matrix_factorizer.inverse_transform(Z)
        Y_hat = XY_hat[:, self.X_n_features:]
        return Y_hat


class RPCA(object):
    n_instances = 0
    r = pyper.R(use_pandas='True', use_numpy=True)

    def __init__(self, n_components=2, method_name='ppca', scale='none', center='FALSE'):
        self.identifier = 'py_{}'.format(self.n_instances)
        self.n_instances += 1
        self.pca_str = 'pca_{}'.format(self.identifier)
        self.loadings_str = 'loadings_{}'.format(self.identifier)
        self.n_components = n_components
        self.method_name = method_name
        self.scale = scale
        self.center = center
        self.loadings = None
        self.scores = None
        self.r('library(pcaMethods)')

    def fit(self, X):
        self._fit(X)

    def _fit(self, X):
        self._fit_transform(X)

    def fit_transform(self, X, filtre=None):
        return self._fit_transform(X, filtre)

    def _fit_transform(self, X, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        F = filtre
        X_str = 'X_{}'.format(self.identifier)
        scores_str = 'scores_{}'.format(self.identifier)
        F_str = 'F_{}'.format(self.identifier)
        self.r.assign(X_str, X)
        self.r.assign(F_str, F)
        self.r('{}[{}==NA] <- NA'.format(X_str, F_str))
        self.r('{} <- pca({}, method="{}", nPcs={}, scale="{}", center={})'.format(self.pca_str, X_str, self.method_name, self.n_components, self.scale, self.center))
        self.r('{} <- {}@loadings'.format(self.loadings_str, self.pca_str))
        self.r('{} <- {}@scores'.format(scores_str, self.pca_str))
        scores = self.r.get(scores_str)
        self.scores = scores
        loadings = self.r.get('t({})'.format(self.loadings_str))
        self.loadings = loadings
        return scores

    def transform(self, X, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        F = filtre
        X_new_str = 'X_new_{}'.format(self.identifier)
        F_str = 'F_{}'.format(self.identifier)
        predict_str = 'predict_{}'.format(self.identifier)
        new_scores_str = 'new_scores_{}'.format(self.identifier)
        self.r.assign(X_new_str, X)
        self.r.assign(F_str, F)
        self.r('{}[{}==NA] <- NA'.format(X_new_str, F_str))
        self.r('{} <- predict({}, {})'.format(predict_str, self.pca_str, X_new_str))
        self.r('{} <- {}$scores'.format(new_scores_str, predict_str))
        new_scores = self.r.get(new_scores_str)
        self.scores = new_scores
        return new_scores

    def inverse_transform(self, scores):
        return scores @ self.loadings
