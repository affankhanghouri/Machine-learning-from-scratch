import numpy as np

class My_Custom_DecisionTreeClassifier:



    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None



    # Gini Impurity = 1 - Σ(p_i)^2
    def _gini_index(self, y):
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity
    


    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float('inf')
        best_splits = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                # Weighted gini
                gini_left = self._gini_index(y[left_mask])
                gini_right = self._gini_index(y[right_mask])
                gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = {
                        'left_X': X[left_mask],
                        'left_y': y[left_mask],
                        'right_X': X[right_mask],
                        'right_y': y[right_mask]
                    }

        return best_feature, best_threshold, best_splits
    



    def _build_tree(self, X, y, depth):
        # Base case
        if len(np.unique(y)) == 1:
            return {'leaf': True, 'class': y[0]}
        
        if depth >= self.max_depth or len(y) == 0:
            # Return most common class
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        feature, threshold, splits = self._best_split(X, y)

        if splits is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        left_subtree = self._build_tree(splits['left_X'], splits['left_y'], depth + 1)
        right_subtree = self._build_tree(splits['right_X'], splits['right_y'], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }



    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)



    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
        



    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x, self.tree) for x in X])
