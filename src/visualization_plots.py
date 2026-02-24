import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Set output directory two levels up from this script (src/../plots)
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")

def plot_roc_curve(y_true, y_probs):
    """Plots and saves the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()
    print(f"Saved ROC curve to {OUTPUT_DIR}/roc_curve.png")

def plot_feature_importance(model, feature_names):
    """Plots and saves feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    print(f"Saved feature importance to {OUTPUT_DIR}/feature_importance.png")

def plot_metrics_bar(y_true, y_pred):
    """Plots and saves precision, recall, and F1 score."""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [prec, rec, f1]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(metrics, values, color=['#4287f5', '#42f560', '#f542a4'])
    plt.ylim(0, 1.1)
    plt.title('Model Performance Metrics')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_bar.png"))
    plt.close()
    print(f"Saved metrics bar chart to {OUTPUT_DIR}/metrics_bar.png")

def plot_class_distribution(y_train, y_test):
    """Plots and saves class distribution comparison."""
    train_counts = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts(normalize=True)
    
    df = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts
    }).fillna(0)
    
    df.plot(kind='bar', figsize=(8, 6))
    plt.title('Class Distribution (Train vs Test)')
    plt.xlabel('Class Label')
    plt.ylabel('Proportion')
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()
    print(f"Saved class distribution plot to {OUTPUT_DIR}/class_distribution.png")

def plot_precision_recall_curve(y_true, y_probs):
    """Plots and saves the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"))
    plt.close()
    print(f"Saved PR curve to {OUTPUT_DIR}/pr_curve.png")

def compare_train_test_accuracy(model, X_train, y_train, X_test, y_test):
    """Plots and saves training vs testing accuracy comparison."""
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    plt.figure(figsize=(5, 5))
    bars = plt.bar(['Train Acc', 'Test Acc'], [train_acc, test_acc], color=['green', 'orange'])
    plt.ylim(0, 1.1)
    plt.title('Training vs Testing Accuracy')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.savefig(os.path.join(OUTPUT_DIR, "train_vs_test_acc.png"))
    plt.close()
    print(f"Saved accuracy comparison to {OUTPUT_DIR}/train_vs_test_acc.png")
