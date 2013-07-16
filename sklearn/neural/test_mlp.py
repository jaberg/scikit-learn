import numpy as np
from mlp import MLPClassifier

def test_auto_8():

    X = np.eye(8)
    y = range(4) + range(4)

    model = MLPClassifier(3, #iscale=.001,
            random_state=123)
    model.fit(X, y)
    p = model.predict(X)

    print p
    print y

    assert np.all(p == y)


