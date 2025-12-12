import numpy as np
import pickle


class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=None, lambda_=0.01, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.cost_history = []
        
        if regularization not in [None, 'l1', 'l2']:
            raise ValueError("regularization must be 'l1', 'l2', or None")
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y):
        n_samples = X.shape[0]
        
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        epsilon = 1e-15
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        
        cost = -(1 / n_samples) * np.sum(
            y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted)
        )
        
        if self.regularization == 'l2':
            l2_penalty = (self.lambda_ / (2 * n_samples)) * np.sum(self.weights ** 2)
            cost += l2_penalty
            
        elif self.regularization == 'l1':
            l1_penalty = (self.lambda_ / n_samples) * np.sum(np.abs(self.weights))
            cost += l1_penalty
        
        return cost
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        for iteration in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            if self.regularization == 'l2':
                dw += (self.lambda_ / n_samples) * self.weights
                
            elif self.regularization == 'l1':
                dw += (self.lambda_ / n_samples) * np.sign(self.weights)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if iteration % 10 == 0:
                cost = self._compute_cost(X, y)
                self.cost_history.append(cost)
                
                if self.verbose and iteration % 100 == 0:
                    print(f"Iteration {iteration}/{self.n_iterations} - Cost: {cost:.4f}")
        
        if self.verbose:
            print(f"Training completed! Final cost: {self.cost_history[-1]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        if self.weights is None:
            raise RuntimeError("Model belum di-train. Call fit() terlebih dahulu.")
        
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'lambda_': self.lambda_,
            'cost_history': self.cost_history
        }
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self):
        return (f"LogisticRegression(learning_rate={self.learning_rate}, "
                f"n_iterations={self.n_iterations}, "
                f"regularization={self.regularization}, "
                f"lambda_={self.lambda_})")
