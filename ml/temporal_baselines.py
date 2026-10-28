import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from ml.temporal_dataset import ConversationSequenceDataset, get_aggregated_features_for_lr

def calculate_full_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.5
        
    acc = float(accuracy_score(y_true, y_pred))
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "fpr": fpr,
        "fnr": fnr
    }

# ---------------------------------------------------------
# Baseline A: Majority Class
# ---------------------------------------------------------
class MajorityClassBaseline:
    def __init__(self):
        self.majority_prob = 0.5
        
    def fit(self, y_train):
        self.majority_prob = float(np.mean(y_train))
        
    def predict_proba(self, n_samples):
        return np.full(n_samples, self.majority_prob, dtype=np.float32)

# ---------------------------------------------------------
# Baseline B: Simple Tactic-Count / Threshold Rule
# ---------------------------------------------------------
class TacticCountRuleBaseline:
    def __init__(self, tactic_thresh=0.2):
        self.tactic_thresh = tactic_thresh
        
    def predict_proba(self, dataset: ConversationSequenceDataset):
        probs = []
        for item in dataset.samples:
            turn_probs = item.get("features", item.get("turn_probabilities", []))
            turn_probs = np.array(turn_probs, dtype=np.float32)
            if len(turn_probs) == 0:
                probs.append(0.0)
            else:
                # Count total tactic detections across all turns
                tactic_activations = (turn_probs > self.tactic_thresh).sum()
                # Normalize count into a simple surrogate probability
                norm_score = min(1.0, float(tactic_activations) / 3.0)
                probs.append(norm_score)
        return np.array(probs, dtype=np.float32)

# ---------------------------------------------------------
# Baseline C: Logistic Regression on Aggregated Features
# ---------------------------------------------------------
class AggregatedLogisticRegressionBaseline:
    def __init__(self, C=1.0, seed=42):
        self.model = LogisticRegression(C=C, random_state=seed, max_iter=1000)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
