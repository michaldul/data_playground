import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from data.dataframes import load_iris


def calc_entropy(y):
    """
    >>> np.isclose(calc_entropy(np.array([1, 2, 3])), -np.log2(1/3))
    True
    """
    entropy = 0
    for cls in np.unique(y):
        prob = (y == cls).sum() / len(y)
        entropy -= prob * np.log2(prob)
    return entropy


def calc_gini(y):
    """
    >>> calc_gini(np.array([1, 2, 2, 2]))
    0.375
    """
    gini = 1
    for cls in np.unique(y):
        prob = (y == cls).sum() / len(y)
        gini -= prob ** 2
    return gini


def information_gain(parent, child_a, child_b, criterion):
    """
    >>> information_gain(np.array([1, 2, 3]), np.array([1, 2]), np.array([3]), calc_entropy) > 0
    True
    >>> information_gain(np.array([1, 1, 2, 2]), np.array([1, 2]), np.array([1, 2]), calc_gini) < 0
    True
    """
    children_entropy = len(child_a) / len(parent) * criterion(child_a) \
                       + len(child_b) * len(parent) * criterion(child_b)
    return calc_entropy(parent) - children_entropy


def find_dominant_class(y):
    """
    >>> find_dominant_class(pd.Series(['setosa', 'virginica', 'virginica']))
    'virginica'
    """
    return y.value_counts().idxmax()


class DecisionTree:
    def __init__(self, max_depth=3, criterion=calc_entropy):
        self.tree = None
        self.max_depth = max_depth
        self.criterion = criterion

    def _generate_splits(self, X, y, depth):
        if depth == 1:
            return {'class': find_dominant_class(y), 'leaf_size': len(y)}

        info_gains = {}
        for feature in X.columns:
            split = X[feature].mean()
            info_gains[feature] = information_gain(y, y[X[feature] < split], y[X[feature] >= split], self.criterion)

        best_feature = max(info_gains, key=info_gains.get)
        best_split = X[best_feature].mean()
        split_mask = X[best_feature] < best_split

        return {'split_feature': best_feature,
                'split_value': best_split,
                'left_node': self._generate_splits(X[split_mask], y[split_mask], depth - 1),
                'right_node': self._generate_splits(X[~split_mask], y[~split_mask], depth - 1)}

    def fit(self, X, y):
        self.tree = self._generate_splits(X, y, self.max_depth)
        print(self.tree)
        return self

    def predict(self, X):
        def _predict(sample, subtree):
            if 'class' in subtree:
                return subtree['class']
            elif sample[subtree['split_feature']] < subtree['split_value']:
                return _predict(sample, subtree['left_node'])
            else:
                return _predict(sample, subtree['right_node'])

        return X.apply(lambda x: _predict(x, self.tree), axis=1)


if __name__ == '__main__':
    iris = load_iris()

    X = iris[[f for f in iris.columns if f != 'class']]
    y = iris['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    cls = DecisionTree(max_depth=4, criterion=calc_entropy).fit(X_train, y_train)
    predictions = cls.predict(X_test)

    print(accuracy_score(y_test, predictions))  # 0.894736842105
