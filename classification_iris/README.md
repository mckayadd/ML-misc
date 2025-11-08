# Iris Classification (Random Forest vs XGBoost)

This project compares **Random Forest** and **XGBoost** classifiers on the classic Iris dataset using a clean, modular Python structure. It demonstrates model training, evaluation, and hyperparameter optimization with `GridSearchCV`.

## Setup
conda create -n ML-misc python=3.12  
conda activate ML-misc  
conda install -c conda-forge scikit-learn xgboost matplotlib pandas  

## Run
Run Random Forest:  
python main.py --model rf  

Run XGBoost:  
python main.py --model xgb  

Run with Grid Search:  
python main.py --model rf --grid  

## Example Output
Best params: {'max_depth': 3, 'max_features': 2, 'n_estimators': 100}  
Best cross-val accuracy: 0.958  
Test Accuracy (best model): 0.967  

## Project Structure
classification_iris/  
├─ src/  
│  ├─ data.py  
│  ├─ metrics.py  
│  └─ models/  
│     ├─ random_forest.py  
│     └─ xgboost.py  
└─ main.py  

## Notes
- Random Forest uses **bagging** (independent trees).  
- XGBoost uses **boosting** (sequential trees correcting previous errors).  
- Both evaluated using accuracy, precision, recall, and F1-score.  


