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



experiment_id = mlflow.set_experiment("Short Model Experiments")

#artifact_location=Path.cwd().joinpath("mlruns").as_uri()



mlflow.sklearn.autolog()

# tags = {"version": "v1", "priority": "P1"}

# mlflow.set_experiment_tags(tags)


# lr = LogisticRegression(solver='liblinear')
# lda= LinearDiscriminantAnalysis()
# svm = SVC(gamma='scale')
# knn = KNeighborsClassifier()
# ada = AdaBoostClassifier(random_state=0)
# gb = GradientBoostingClassifier(random_state=0)
rf = RandomForestClassifier(random_state=0)
et=  ExtraTreesClassifier(random_state=0)
xgbc = XGBClassifier(random_state=0)


#models = [lr,lda,svm,knn,ada,gb,rf,et,xgbc]
models = [rf,et,xgbc]

for model in models: 
    with mlflow.start_run(run_name=f'Run {model}') as run:
        
        # mlflow.set_tag("model", model)
        # mlflow.set_tag("data scientist", "kb")
        X= df.drop('HeartDisease', axis=1)
        y= df['HeartDisease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        ohe= OneHotEncoder()
        ct= make_column_transformer((ohe,categorical),remainder='passthrough') 
        pipe = make_pipeline(ct, model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred),4)
        #mlflow.log_metric("accuracy", accuracy)
        #mlflow.sklearn.log_model(model, "model")
        #mlflow.log_artifact(model.b)