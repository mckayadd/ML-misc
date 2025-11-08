from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, X_test, y_test, labels=None, label_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred, labels=labels))
    return acc
