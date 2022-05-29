import pandas as pd
import numpy as np

from scripts import log
import os
import dvc.api
import sys
sys.path.append(os.path.abspath(os.path.join('data')))
sys.path.insert(0,'../scripts/')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from scripts.datacleaner import DataCleaner

cleaner = DataCleaner()

logger = log.setup_custom_logger(
    __name__, file_name='./logs/feature_engineering.log')


class FeatureEngineering:

    def __init__(self, df):
        self.df = df

    def preprocess(self):
        preprocess_pipeline = Pipeline(steps=[
            ('add_date_features', FunctionTransformer(self.add_date_features, validate=False))       

        ])
        self.df = preprocess_pipeline.fit_transform(self.df)


    def calc_date_gap(self, holidays: list, day) -> tuple:
        '''
            Calculate the gap between the given date and the next and previous holiday
        '''
        after = 366
        after_date = None
        before = 366
        before_date = None
        for holiday in holidays:
            if holiday > day:
                distance = (holiday - day).days
                if distance < after:
                    after = distance
                    after_date = holiday
            else:
                distance = np.abs((holiday - day).days)
                if distance < before:
                    before = distance
                    before_date = holiday

        return (before, before_date), (after, after_date)

    def get_holidays(self, df: pd.DataFrame) -> list:
        '''
            Get the list of holidays from the dataframe
        '''
        holidays = None
        holidays = df[df['StateHoliday'] != '0']['Date'].unique()
        holidays.sort()
        return holidays

    def time_in_month(self, day: int) -> int:
        """
            Calculate the time in month
            From 1 to 10th day -> 0
            From 11 to 20th day -> 1
            From 21 to 31th day -> 2
        """
        try:
            day = int(day)
            if(1 < day < 10):
                return 0
            elif(10 <= day < 20):
                return 1
            else:
                return 2
        except ValueError:
            logger.error("Invalid day")

    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
            Generate new data columns using the existing date column
        '''
        df['Date'] = pd.to_datetime(df['Date'])
        df['weekday'] = df['DayOfWeek'].apply(lambda x: 1 if x < 6 else 0)
        df['weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        df['month'] = df['Date'].apply(lambda x: x.month)
        df['year'] = df['Date'].apply(lambda x: x.year)
        df['day'] = df['Date'].apply(lambda x: x.day)
        df['period_in_month'] = df['Date'].apply(
            lambda x: self.time_in_month(x.day))
        holidays = self.get_holidays(df)
        before_list = []
        after_list = []
        for ind in df.index:
            before, after = self.calc_date_gap(holidays, df['Date'][ind])
            if before[0] != 366:
                before_list.append(before[0])
            else:
                before_list.append(0)
            if after[0] != 366:
                after_list.append(after[0])
            else:
                after_list.append(0)
        df['before_holiday'] = before_list
        df['after_holiday'] = after_list
        df[['weekday', 'weekend',  'period_in_month']] = df[['weekday', 'weekend',  'period_in_month']].astype("category")
        logger.info('8 Columns added to the dataframe.')

        return df
