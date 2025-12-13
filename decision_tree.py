"""
Decision Tree Learning (CART) - Implementation from Scratch

This module implements the Classification and Regression Tree (CART) algorithm
for binary classification, specifically designed for fraud detection.

Features:
- Gini impurity as splitting criterion
- Handles numerical and categorical (encoded) features
- Handles missing values (NaN)
- Supports max_depth, min_samples_split, min_samples_leaf
- Save/Load functionality with pickle

Author: Tubes DAI Team
"""

import numpy as np
import pandas as pd
import pickle


class DecisionTreeNode:
    """
    Node for Decision Tree.
    
    Attributes:
        feature_index: Index of feature to split on
        threshold: Threshold value for split
        left: Left child node
        right: Right child node
        value: Class prediction (for leaf nodes)
        is_leaf: Whether this is a leaf node
        gini: Gini impurity at this node
        samples: Number of samples at this node
    """
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, 
                 value=None, is_leaf=False, gini=None, samples=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf
        self.gini = gini
        self.samples = samples


class CARTDecisionTree:
    """
    Classification and Regression Tree (CART) Decision Tree Classifier.
    
    Implementation from scratch that handles:
    - Numerical features
    - Categorical features (encoded)
    - Missing values (handled in preprocessing)
    
    Uses Gini impurity as the splitting criterion.
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a split.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    root : DecisionTreeNode
        The root node of the tree.
    n_features : int
        Number of features in training data.
    feature_names : list
        Names of features (if provided as DataFrame).
    classes_ : ndarray
        Unique class labels.
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 min_impurity_decrease=0.0, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root = None
        self.n_features = None
        self.feature_names = None
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels.
        
        Gini = 1 - sum(p_i^2) for all classes i
        
        Parameters:
        -----------
        y : array-like
            Target labels
            
        Returns:
        --------
        float
            Gini impurity value
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini
    
    def _gini_gain(self, y, y_left, y_right):
        """
        Calculate the weighted Gini gain from a split.
        
        Gain = Gini(parent) - weighted average of Gini(children)
        
        Parameters:
        -----------
        y : array-like
            Parent node labels
        y_left : array-like
            Left child labels
        y_right : array-like
            Right child labels
            
        Returns:
        --------
        float
            Information gain
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_gini = self._gini_impurity(y)
        left_gini = self._gini_impurity(y_left)
        right_gini = self._gini_impurity(y_right)
        
        weighted_child_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        gain = parent_gini - weighted_child_gini
        
        return gain
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray
            Target labels
            
        Returns:
        --------
        tuple
            (best_feature_index, best_threshold, best_gain)
        """
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Get unique values and sort them
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= 1:
                continue
            
            # Try thresholds between consecutive unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate gain
                gain = self._gini_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray
            Target labels
        depth : int
            Current depth in tree
            
        Returns:
        --------
        DecisionTreeNode
            Root node of (sub)tree
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Calculate current node Gini
        current_gini = self._gini_impurity(y)
        
        # Determine majority class for this node
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        
        # Stopping conditions - create leaf node
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return DecisionTreeNode(
                value=majority_class,
                is_leaf=True,
                gini=current_gini,
                samples=n_samples
            )
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found or gain is too small, create leaf
        if best_feature is None or best_gain < self.min_impurity_decrease:
            return DecisionTreeNode(
                value=majority_class,
                is_leaf=True,
                gini=current_gini,
                samples=n_samples
            )
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gini=current_gini,
            samples=n_samples
        )
    
    def fit(self, X, y):
        """
        Build the decision tree from training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self
            Fitted estimator
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        # Handle NaN values (replace with column median)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                median_val = np.nanmedian(X[:, i])
                X[mask, i] = median_val
        
        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        return self
    
    def _predict_single(self, x, node):
        """
        Predict class for a single sample by traversing the tree.
        
        Parameters:
        -----------
        x : ndarray
            Single sample features
        node : DecisionTreeNode
            Current node
            
        Returns:
        --------
        int or float
            Predicted class label
        """
        if node.is_leaf:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X, dtype=np.float64)
        
        # Handle NaN values
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X[mask, i] = 0  # Replace NaN with 0 for prediction
        
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Note: This simple implementation returns hard probabilities (0 or 1).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        predictions = self.predict(X)
        n_samples = len(predictions)
        n_classes = len(self.classes_)
        
        proba = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba
    
    def get_depth(self, node=None):
        """
        Get the depth of the tree.
        
        Parameters:
        -----------
        node : DecisionTreeNode, optional
            Starting node (default: root)
            
        Returns:
        --------
        int
            Tree depth
        """
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return 0
        
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))
    
    def get_n_leaves(self, node=None):
        """
        Get the number of leaf nodes in the tree.
        
        Parameters:
        -----------
        node : DecisionTreeNode, optional
            Starting node (default: root)
            
        Returns:
        --------
        int
            Number of leaf nodes
        """
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return 1
        
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """
        Print the tree structure.
        
        Parameters:
        -----------
        node : DecisionTreeNode, optional
            Starting node (default: root)
        depth : int
            Current depth for indentation
        prefix : str
            Label for current node
        """
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf:
            print(f"{indent}{prefix}: Leaf - Class={node.value}, Gini={node.gini:.4f}, Samples={node.samples}")
        else:
            feature_name = self.feature_names[node.feature_index] if self.feature_names else f"Feature_{node.feature_index}"
            print(f"{indent}{prefix}: {feature_name} <= {node.threshold:.4f}, Gini={node.gini:.4f}, Samples={node.samples}")
            self.print_tree(node.left, depth + 1, "Left")
            self.print_tree(node.right, depth + 1, "Right")
    
    def save_model(self, filepath):
        """
        Save the model to a file using pickle.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        CARTDecisionTree
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Simple test
    print("CART Decision Tree - Test")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Train model
    model = CARTDecisionTree(max_depth=5, random_state=42)
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Tree Depth: {model.get_depth()}")
    print(f"Number of Leaves: {model.get_n_leaves()}")