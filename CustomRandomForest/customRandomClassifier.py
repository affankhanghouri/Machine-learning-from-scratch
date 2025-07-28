import numpy as np
from collections import Counter

class SimpleDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            most_common = Counter(y).most_common(1)[0][0]
            return most_common

        # Find best split
        best_feat, best_thresh = self.best_split(X, y)
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        # Split
        left_idx = X[:, best_feat] < best_thresh
        right_idx = X[:, best_feat] >= best_thresh

        left_tree = self.fit(X[left_idx], y[left_idx], depth + 1)
        right_tree = self.fit(X[right_idx], y[right_idx], depth + 1)

        return (best_feat, best_thresh, left_tree, right_tree)

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left = y[X[:, feature] < thresh]
                right = y[X[:, feature] >= thresh]
                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self.information_gain(y, left, right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh

        return split_idx, split_thresh

    def entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def information_gain(self, parent, left, right):
        weight_l = len(left) / len(parent)
        weight_r = len(right) / len(parent)
        return self.entropy(parent) - (weight_l * self.entropy(left) + weight_r * self.entropy(right))

    def predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, thresh, left, right = tree
        branch = left if x[feature] < thresh else right
        return self.predict_one(x, branch)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

    def train(self, X, y):
        self.tree = self.fit(X, y)




#_-------------------------------------------


class MyRandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority vote
        majority_preds = [Counter(preds).most_common(1)[0][0] for preds in all_preds.T]
        return np.array(majority_preds)
