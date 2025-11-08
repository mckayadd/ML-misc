# src/data.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # <-- make sure this import exists

def load_iris_split(test_size: float = 0.2, seed: int = 42):
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    # return the 4 splits + the label names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    return (X_train, X_test, y_train, y_test), iris.target_names
