import pandas as pd
import dvc.api

# To Split our train data
from sklearn.model_selection import train_test_split

# To Preproccesing our data
from sklearn.preprocessing import LabelEncoder

from scripts import log

logger = log.setup_custom_logger(__name__, file_name='../logs/utils.log')

class Utils:
    '''
    Utility class for preprocessing data.
    '''
    def load_data_dvc(self,tag:str, data_path: str, repo:str) -> pd.DataFrame:
        """
        Load data from a csv file.
        """
        try:
            with dvc.api.open(
                repo=repo, 
                path=data_path, 
                rev=tag,
                mode="r"
            ) as fd:
                df = pd.read_csv(fd)
                logger.info("Data loaded from DVC.")
        except Exception:
            logger.error("File not found.")
        return df

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from a csv file.
        """
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            logger.error("File not found.")
        return df

    def save_csv(self, df:pd.DataFrame, csv_path:str):
        """
        Save data to a csv file.
        """
        try:
            df.to_csv(csv_path, index=False)
            logger.info('File Successfully Saved.!!!')
        except Exception:
            logger.error("Save failed...")
        return df

    def split_train_test_val(self, input_data:tuple, size:tuple)-> list:
        """
        Split the data into train, test and validation.
        """
        X,Y = input_data
        train_x, temp_x, train_y, temp_y = train_test_split(X, Y, train_size=size[0], test_size=size[1]+size[2], random_state=42)
        test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y, train_size=size[1]/(size[1]+size[2]), test_size=size[2]/(size[1]+size[2]), random_state=42)
        return [train_x, train_y, test_x, test_y, val_x, val_y]

    def encode_features(self,df:pd.DataFrame)-> pd.DataFrame:
        """
        Encode features using LabelEncoder.
        """
        features = df.columns
        for feature in features:
            le = LabelEncoder()
            le.fit(df[feature])
            df[feature] = le.transform(df[feature])
        return df
        