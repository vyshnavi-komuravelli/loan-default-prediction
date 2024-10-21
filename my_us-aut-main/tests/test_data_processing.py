import os
import sys
from unittest import TestCase

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))

from src.customer_analysis import preprocessing1

from pathlib import Path
BASE_DIR = Path('.').resolve().parent
print(BASE_DIR)
car_type=pd.read_csv(BASE_DIR/"data"/"Car_Type.csv")
caov2=pd.read_csv(BASE_DIR/"data"/"close_and_open_version.csv")
open1 = caov2[caov2["Next_Qtr_Account_Status"].isnull()]
open1=open1[open1['Qtr_Account_Status']=='OPEN']
closed = caov2[caov2["Next_Qtr_Account_Status"].notnull()]

class Testingest(TestCase):
    def test_is_df(self):
        train_set,test_set=preprocessing1(closed,open1,car_type)
        assert isinstance(train_set, pd.DataFrame)
        assert isinstance(test_set, pd.DataFrame)
