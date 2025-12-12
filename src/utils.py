import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)


def standardize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def handle_missing_values(X, strategy='mean'):
    X_copy = X.copy()
    
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        
        if np.any(mask):
            if strategy == 'mean':
                fill_value = np.nanmean(X[:, col])
            elif strategy == 'median':
                fill_value = np.nanmedian(X[:, col])
            elif strategy == 'mode':
                non_nan = X[~mask, col]
                if len(non_nan) > 0:
                    fill_value = np.bincount(non_nan.astype(int)).argmax()
                else:
                    fill_value = 0
            else:
                fill_value = 0
            
            X_copy[mask, col] = fill_value
    
    return X_copy


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='weighted', zero_division=0):
    classes = np.unique(y_true)
    precisions = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        
        if tp + fp == 0:
            precision = zero_division
        else:
            precision = tp / (tp + fp)
        
        precisions.append(precision)
        weights.append(np.sum(y_true == cls))
    
    if average == 'weighted':
        return np.average(precisions, weights=weights)
    elif average == 'macro':
        return np.mean(precisions)
    else:
        return precisions


def recall_score(y_true, y_pred, average='weighted', zero_division=0):
    classes = np.unique(y_true)
    recalls = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        if tp + fn == 0:
            recall = zero_division
        else:
            recall = tp / (tp + fn)
        
        recalls.append(recall)
        weights.append(np.sum(y_true == cls))
    
    if average == 'weighted':
        return np.average(recalls, weights=weights)
    elif average == 'macro':
        return np.mean(recalls)
    else:
        return recalls


def f1_score(y_true, y_pred, average='weighted', zero_division=0):
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    
    if precision + recall == 0:
        return zero_division
    
    return 2 * (precision * recall) / (precision + recall)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def print_metrics(metrics, model_name):
    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
