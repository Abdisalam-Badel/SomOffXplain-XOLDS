from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score

class ClassificationMetrics:
    def __init__(self, class_names):
        self.class_names = class_names
    def __call__(self, y_true, y_pred, y_probs=None):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        report = classification_report(
            y_true, y_pred,
            target_names=[self.class_names[0], self.class_names[1]],
            digits=4, zero_division=0
        )
        auroc = roc_auc_score(y_true, y_probs) if y_probs is not None else None
        return acc, prec, rec, f1, report, auroc
