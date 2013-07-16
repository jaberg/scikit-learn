# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# Author: James Bergstra

import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..preprocessing import LabelBinarizer
from ..utils import atleast2d_or_csr, check_random_state
from ..utils.extmath import logsumexp, safe_sparse_dot


def logistic(x):
    return 1. / (1. + np.exp(-x))


def log_softmax(X):
    # Computes the logistic K-way softmax, (exp(X).T / exp(X).sum(axis=1)).T,
    # in the log domain
    return (X.T - logsumexp(X, axis=1)).T


def softmax(X):
    if X.shape[1] == 1:
        return logistic(X)
    else:
        exp_X = np.exp(X)
        return (exp_X.T / exp_X.sum(axis=1)).T


def _tanh(X):
    """Hyperbolic tangent with LeCun's magic constants."""
    X *= 2/3.
    np.tanh(X, X)
    X *= 1.7159
    return X


class MLPSGD(object):
    """
    Optimizer:
    Algorithm: 'sgd'
        batch_size : int, optional
            Size of minibatches in SGD optimizer.
        learning_rate : float, optional
            Base learning rate. This will be scaled by sqrt(n_features) for the
            input-to-hidden weights, and by sqrt(n_hidden) for the hidden-to-output
            weights.
        max_iter : int, optional
            Maximum number of iterations.
        momentum : float, optional
            Parameter for the momentum method. Set this somewhere between .5 and 1.
        shuffle : bool, optional
            Whether to shuffle samples in each iteration before extracting
            minibatches.
        tol : float, optional
            Tolerance for the optimization. When the loss at iteration i+1 differs
            less than this amount from that at iteration i, convergence is
            considered to be reached.
    """
    def __init__(self, batch_size=100,
                 learning_rate=.01, max_iter=100, momentum=.9,
                 shuffle=True, tol=1e-5,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.momentum = momentum
        self.tol = tol
        self.shuffle = shuffle

    def fit_model(self, mlp, X, Y):
        NX, DX = X.shape
        NY, = Y.shape
        assert NX == NY, (NX, NY)
        for epoch in range(2000):
            losses = []
            for ii in range(0, len(X), self.batch_size):
                #print '-' * 80
                #print 'ii', ii, self.learning_rate
                Xii = X[ii:ii + self.batch_size]
                Yii = Y[ii:ii + self.batch_size]
                gparams = mlp.params_.zeros_like()
                loss = mlp.decision_function_grad(Xii, Yii, gparams,
                        pcoef=0.0, gcoef=1.0)
                mlp.params_.params -= self.learning_rate * gparams.params
                losses.append(loss)
            print np.mean(losses)


class MLPParams(object):
    def __init__(self, n_features, n_hidden, n_targets, dtype='float'):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_targets = n_targets
        param_shapes = [(n_hidden, n_features),
                        (n_targets, n_hidden),
                        (n_hidden,),
                        (n_targets,)]

        self.params = np.zeros(sum(map(np.prod, param_shapes)), dtype=dtype)
        self.views = []
        offset = 0
        for shape in param_shapes:
            self.views.append(self.params[offset:offset + np.prod(shape)].reshape(shape))
            offset += np.prod(shape)

    def __str__(self):
        return "w: %s\nb: %s\nv: %s\nc:%s" % (
                self.coef_hidden,
                self.intercept_hidden,
                self.coef_output,
                self.intercept_output)

    def zeros_like(self):
        return MLPParams(self.n_features, self.n_hidden, self.n_targets,
                dtype=self.params.dtype)

    @property
    def coef_hidden(self):
        return self.views[0]

    @coef_hidden.setter
    def coef_hidden(self, value):
        self.views[0][:] = value

    @property
    def coef_output(self):
        return self.views[1]

    @coef_output.setter
    def coef_output(self, value):
        self.views[1][:] = value

    @property
    def intercept_hidden(self):
        return self.views[2]

    @intercept_hidden.setter
    def intercept_hidden(self, value):
        self.views[2][:] = value

    @property
    def intercept_output(self):
        return self.views[3]

    @intercept_output.setter
    def intercept_output(self, value):
        self.views[3][:] = value


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Multi-layer perceptron (feedforward neural network) classifier.

    Trained with gradient descent under log loss (aka the cross-entropy error
    function).

    Parameters
    ----------
    n_hidden : int
        Number of units in the hidden layer.
    activation: string, optional
        Activation function for the hidden layer; either "logistic" for
        1 / (1 + exp(x)), or "tanh" for the hyperbolic tangent.
    alpha : float, optional
        L2 penalty (weight decay) parameter.
    optimizer: dict, optional
    random_state : int or RandomState, optional
        State of or seed for random number generator.
    verbose : bool, optional
        Whether to print progress messages to stdout.

    """
    def __init__(self, n_hidden, activation="tanh", alpha=0,
                 random_state=None, algo=MLPSGD(), verbose=False,
                 iscale=None, dtype=np.float64):
        self.activation = activation
        self.alpha = alpha
        self.n_hidden = n_hidden
        self.random_state = check_random_state(random_state)
        self.algo = algo
        self.verbose = verbose
        self.iscale = iscale
        self.dtype = dtype

    def _init_fit(self, n_features, n_targets):
        n_hidden = self.n_hidden
        rng = self.random_state

        if self.iscale is None:
            # -- Default advocated by Glorot and Bengio, 2010
            iscale = 6. / np.sqrt(self.n_hidden + n_features)
        else:
            iscale = self.iscale
        self.params_ = MLPParams(n_features, n_hidden, n_targets)

        self.params_.coef_hidden = rng.uniform(-1, 1, (n_hidden, n_features)) * iscale
        self.params_.coef_output = np.zeros((n_targets, n_hidden))

        self.params_.intercept_hidden = np.zeros(n_hidden)
        self.params_.intercept_output = np.zeros(n_targets)

    def fit(self, X, y):
        X = atleast2d_or_csr(X, dtype=np.float64, order="C")
        y = np.asarray(y)
        _, n_features = X.shape
        n_targets = np.max(y) + 1

        self._init_fit(n_features, n_targets)
        self.algo.fit_model(self, X, y)

        return self

    def partial_fit(self, X, y, classes):
        X = atleast2d_or_csr(X, dtype=np.float64, order="C")
        _, n_features = X.shape

        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call "
                             "to partial_fit.")
        elif classes is not None and self.classes_ is not None:
            if not np.all(self.classes_ == np.unique(classes)):
                raise ValueError("`classes` is not the same as on last call "
                                 "to partial_fit.")
        elif classes is not None:
            self._lbin = LabelBinarizer(classes=classes)
            Y = self._lbin.fit_transform(y)
            self._init_fit(n_features, Y.shape[1])
        else:
            Y = self._lbin.transform(y)

        self.algo.fit_partial_model(self, X, Y)

        return self

    def decision_function(self, X):
        X = atleast2d_or_csr(X)
        z_hidden = (safe_sparse_dot(X, self.params_.coef_hidden.T) +
                    self.params_.intercept_hidden)
        y_hidden = logistic(z_hidden) if self.activation == "logistic" \
                                      else _tanh(z_hidden)
        y_output = (np.dot(y_hidden, self.params_.coef_output.T) +
                    self.params_.intercept_output)
        print  X
        print y_output
        return y_output

    def decision_function_grad(self, X, Y, gparams, pcoef, gcoef):
        """
        if gparams is None it uses self.params
        """
        X = atleast2d_or_csr(X)
        z_hidden = (safe_sparse_dot(X, self.params_.coef_hidden.T) +
                    self.params_.intercept_hidden)
        y_hidden = logistic(z_hidden) if self.activation == "logistic" \
                                      else _tanh(z_hidden)
        y_output = (np.dot(y_hidden, self.params_.coef_output.T) +
                    self.params_.intercept_output)

        # -- make sure this next bit gets into Cython or numba or something
        max_out = np.max(y_output, axis=1)
        p_output = np.exp(y_output - max_out[:, None])
        p_output /= p_output.sum(axis=1)[:, None]
        g_p_output = p_output.copy()
        g_p_output[range(len(Y)), Y] -= 1.0
        loss = -np.log(p_output[range(len(Y)), Y]) # -- XXX use pre-softmax value for acc.

        g_hid_out = np.dot(g_p_output, self.params_.coef_output)
        if self.activation == 'logistic':
            g_hid_in = g_hid_out * z_hidden * (1 - z_hidden)
        else:
            g_hid_in = g_hid_out * (1 - z_hidden ** 2)

        if pcoef:
            gparams.params[:] = self.params_.params * pcoef
        else:
            gparams.params[:] = 0

        gparams.intercept_output += gcoef * np.mean(g_p_output, axis=0)
        gparams.intercept_hidden += gcoef * np.mean(g_hid_in, axis=0)
        gparams.coef_output += gcoef * np.dot(g_p_output.T, z_hidden)
        gparams.coef_hidden += gcoef * np.dot(g_hid_in.T, X)

        return loss

    def predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_log_proba(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            return np.log(logistic(scores))
        else:
            return log_softmax(scores)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            return logistic(scores)
        else:
            return softmax(scores)

    @property
    def classes_(self):
        return np.arange(self.params_.n_targets)

