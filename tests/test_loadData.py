import unittest
import pandas as pd
import sys
sys.path.append('../')
from scripts.utils import Utils
from scripts import log

# logger = log.setup_custom_logger(__name__, file_name='logs/DataLoaderTest.log')
utils = Utils()

class TestDataLoader(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        pass
        

    def test_read_csv(self):
        ad_df = utils.load_data('tests/test_df.csv')
        self.assertEqual(len(ad_df), 6)
        print("Reading local file successful")
    
    def test_dvc_get_data(self):
        data_path = 'test/test_df.csv'
        tag = 'v3-test'
        repo = '.'
        test_df = utils.load_data_dvc(tag,data_path, repo)
        self.assertTrue(test_df['Open'].isnull().any())
        print("Reading data from dvc successfull")


if __name__ == '__main__':
    unittest.main()