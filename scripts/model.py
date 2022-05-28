import os
import sys
from urllib.parse import urlparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression


from datacleaner import DataCleaner
from utils import Utils

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import log

import mlflow
sys.path.append(os.path.abspath(os.path.join('data')))
sys.path.insert(0, '../scripts/')



utils = Utils()
cleaner = DataCleaner()

logger = log.setup_custom_logger(
    __name__, file_name='../logs/model.log')


class Model:
    def __init__(self, data, model) -> None:
        self.data = data
        self.model = model
        self.preprocessed_data = None

    def split_target_feature(self, df: pd.DataFrame, target_col: str) -> tuple:

        target = df[[target_col]]
        features = df.drop(target_col, axis=1)
        return features, target

    def preprocess(self):
        preprocess_pipeline = Pipeline(steps=[
            ('label_encoder', FunctionTransformer(
                cleaner.encode_features, validate=False)),
            ('scaler', FunctionTransformer(cleaner.standard_scaler, validate=False)),
            ('target_feature_split', FunctionTransformer(self.split_target_feature, kw_args={'target_col':'Sales'}, validate=False)),
            ('train_test_split', FunctionTransformer(utils.split_train_test_val,kw_args={'size':(.7,.2,.1)}, validate=False))
        ])
        logger.info("Preprocessing complete. Data ready for modeling.")
        return preprocess_pipeline.fit_transform(self.data)

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def train(self, args, exp_name):
        mlflow.set_experiment(exp_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run():
            self.preprocessed_data = self.preprocess()
            model = self.model(**args)
            fitted_model = model.fit(self.preprocessed_data[0], self.preprocessed_data[1])
            y_pred = fitted_model.predict(self.preprocessed_data[2])

            (rmse, mae, r2) = self.eval_metrics(self.preprocessed_data[3], y_pred)
            mlflow.log_param("n_estimators", args['n_estimators'])
            mlflow.log_param("max_features", args['max_features'])
            mlflow.log_param("max_depth", args['max_depth'])
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            result_df = self.preprocessed_data[2].copy()
            result_df["Prediction Sales"] = y_pred
            result_df["Actual Sales"] = self.preprocessed_data[3]
            result_agg = result_df.groupby("day").agg(
            {"Prediction Sales": "mean", "Actual Sales": "mean"})

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(fitted_model, "model", registered_model_name="RandomForestModel")
            else:
                mlflow.sklearn.log_model(fitted_model, "model") 
            
        return fitted_model, result_agg

    def get_features_importance(self,fitted_model):
        # try:
        #     if (isinstance(fitted_model.steps[1][1], LinearRegression)):
        #         model = fitted_model.steps[1][1]

        #         p_df = pd.DataFrame()
        #         p_df['features'] = self.preprocessed_data[0].columns.to_list()
        #         p_df['coff_importance'] = abs(model.coef_)

        #         return p_df
        # except Exception:
        #      logger.error("Model is not a linear regression model")   

        importance = fitted_model.feature_importances_
        f_df = pd.DataFrame(columns=["features", "importance"])
        f_df["features"] = self.preprocessed_data[0].columns.to_list()
        f_df["importance"] = importance
        return f_df

    def prediction_graph(self, res_dataframe):

        fig = plt.figure(figsize=(18, 5))
        sns.lineplot(x=res_dataframe.index,
                     y=res_dataframe["Actual Sales"], label='Actual')
        sns.lineplot(x=res_dataframe.index,
                     y=res_dataframe["Prediction Sales"], label='Prediction')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel="Day", fontsize=16)
        plt.ylabel(ylabel="Sales", fontsize=16)
        plt.show()
