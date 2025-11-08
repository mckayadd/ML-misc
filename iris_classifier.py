import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


###

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, None],
    'max_features': [2, 3, 4]
}

###


iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
df = iris.frame

# print(df.describe)


### GridSearch object

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

###


# 20% test, 80% train, mix randomly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# check
# print("X_train: ", X_train.shape)
# print("X_test: ", X_test.shape)
# print("y train value counts: ", y_train.value_counts().sort_index())
# print("y test value counts: ", y_test.value_counts().sort_index())



# building model
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    random_state=42
)

# train
# rf.fit(X_train, y_train)
grid.fit(X_train, y_train)


# best results
print("Best parameters:", grid.best_params_)
print(f"Best cross-val accuracy: {grid.best_score_:.3f}")

best_rf = grid.best_estimator_
test_acc = best_rf.score(X_test, y_test)
print(f"Test Accuracy (best model): {test_acc:.3f}")

# # prediction
# y_pred = rf.predict(X_test)

# # accuracy
# acc = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {acc:.3f}")

# print("First 10 preds:", y_pred[:10].tolist())


# ################################

# # metrics
# print("\n=== Classification report ===")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))

# # confusion matrix
# cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
# print("\n=== Confusion matrix (raw) ===")
# print(cm)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
# disp.plot(values_format='d')
# plt.title("Random Forest - Confusion Matrix")
# plt.tight_layout()
# plt.show()

# # feature importances
# fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
# print("\n=== Feature importances ===")
# print(fi)

# # visualization
# fi.plot(kind='bar')
# plt.title("Feature Importances (Random Forest)")
# plt.ylabel("Importance")
# plt.tight_layout()
# plt.show()
