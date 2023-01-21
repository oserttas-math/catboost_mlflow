
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
from pathlib import Path

from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import optuna
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,classification_report

# importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# import plotly 
# import plotly.express as px
# import plotly.graph_objs as go
# import plotly.offline as py
# from plotly.offline import iplot
# from plotly.subplots import make_subplots
# import plotly.figure_factory as ff

import missingno as msno

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('heart.csv')


numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns


# with mlflow.start_run():
#     mlflow.log_artifact("models", artifact_path="preprocessor")
#     #mlflow.log_artifact(".")
#     mlflow.set_tag("data scientist", "kb")
#     X= df.drop('HeartDisease', axis=1)
#     y= df['HeartDisease']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     ohe= OneHotEncoder()
#     ct= make_column_transformer((ohe, categorical),remainder='passthrough')  
#     lr = LogisticRegression(solver='liblinear')
#     mlflow.set_tag("model", lr)
#     pipe = make_pipeline(ct, lr)
#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict(X_test)
#     accuracy = round(accuracy_score(y_test, y_pred),4)
#     mlflow.log_metric("accuracy", accuracy)


#os.makedirs("models", exist_ok=True)



experiment_id = mlflow.set_experiment("Catboost Experiments")

#artifact_location=Path.cwd().joinpath("mlruns").as_uri()



#mlflow.sklearn.autolog()





def objective(trial):
    with mlflow.start_run():
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }
        mlflow.set_tag("model", "catboost")
        mlflow.set_tag("data scientist", "kb")
        mlflow.log_params(param)
        X= df.drop('HeartDisease', axis=1)
        y= df['HeartDisease']
        categorical_features_indices = np.where(X.dtypes != np.float)[0]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


        cat_cls = CatBoostClassifier(**param)


        cat_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], cat_features=categorical_features_indices,verbose=0, early_stopping_rounds=100)
        mlflow.catboost.log_model(cat_cls, artifact_path="preprocessor")


        preds = cat_cls.predict(X_test)
        pred_labels = np.rint(preds)
        accuracy = round(accuracy_score(y_test, pred_labels),4)
        mlflow.log_metric("accuracy", accuracy)
        return accuracy




if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)


    print("Number of finished trials: {}".format(len(study.trials)))


    print("Best trial:")
    trial = study.best_trial


    print("  Value: {}".format(trial.value))


    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))