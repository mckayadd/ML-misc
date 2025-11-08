import argparse
from sklearn.model_selection import GridSearchCV
from src.data import load_iris_split
from src.metrics import evaluate
from src.models import random_forest as rf_mod
from src.models import xgboost as xgb_mod

def run(model_name: str, do_grid: bool):
    (X_train, X_test, y_train, y_test), names = load_iris_split()

    if model_name == "rf":
        cfg = rf_mod.RFConfig()
        model = rf_mod.build_model(cfg)
        param_grid = rf_mod.grid()
    elif model_name == "xgb":
        cfg = xgb_mod.XGBConfig()
        model = xgb_mod.build_model(cfg)
        param_grid = xgb_mod.grid()
    else:
        raise ValueError("model must be 'rf' or 'xgb'")
    

    if do_grid:
        gs = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        print("Best params:", gs.best_params_)
        model = gs.best_estimator_
    else:
        model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, labels=[0,1,2], label_names=names)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    run(args.model, args.grid)