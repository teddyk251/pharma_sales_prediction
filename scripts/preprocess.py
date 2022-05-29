import os
import pandas as pd
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join('data')))
sys.path.insert(0, '../scripts/')
from scripts.utils import Utils
from scripts.datacleaner import DataCleaner
from scripts.feature_engineering import FeatureEngineering
from scripts import log

cleaner = DataCleaner()
utils = Utils()

logger = log.setup_custom_logger(__name__, file_name='logs/preprocess_dashboard.log')


class Preprocess:
    def prepare_data(self, data):
        """
        Prepare data for training.
        """
        preprocess_pipeline = Pipeline(steps=[
            ('set_type', FunctionTransformer(self.set_type, validate=False)),
            ('label_encoder', FunctionTransformer(
                cleaner.encode_features, validate=False)),
            ('scaler', FunctionTransformer(cleaner.standard_scaler, validate=False)),
        ])
        return preprocess_pipeline.fit_transform(data)


    def set_type(self, data):
        try:
            data['Store'] = data['Store'].astype('int')
            data['DayOfWeek'] = data['DayOfWeek'].astype('int')
            data['Date'] = pd.to_datetime(data['Date'])
            data['Customers'] = data['Customers'].astype('int')
            data['Open'] = data['Open'].astype('int')
            data['Promo'] = data['Promo'].astype('int')
            data['StateHoliday'] = data['StateHoliday'].astype('object')
            data['SchoolHoliday'] = data['SchoolHoliday'].astype('int')
            data['StoreType'] = data['StoreType'].astype('object')
            data['Assortment'] = data['Assortment'].astype('object')
            data['CompetitionDistance'] = data['CompetitionDistance'].astype('float')
            data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].astype('float')
            data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].astype('float')
            data['Promo2'] = data['Promo2'].astype('int')
            data['Promo2SinceWeek'] = data['Promo2SinceWeek'].astype('float')
            data['Promo2SinceYear'] = data['Promo2SinceYear'].astype('float')
            data['PromoInterval'] = data['PromoInterval'].astype('object')
        except:
            logger.error("Unable to set data type")

        return data
        