import sys
import os

sys.path.insert(0, os.path.abspath(".."))
from src.customer_analysis import (
    preprocessing1,
    training,
    final_model1,
    score,
    preprocessing2
)
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

BASE_DIR = Path(".").resolve().parent.parent

with open("conf/config.yaml","r") as f:
    data=yaml.safe_load(f)
    
car_type = pd.read_csv(data["car_type"])
caov2 = pd.read_csv(data["us_auto"])
open1 = caov2[caov2["Next_Qtr_Account_Status"].isnull()]
open1=open1[open1['Qtr_Account_Status']=='OPEN']
closed = caov2[caov2["Next_Qtr_Account_Status"].notnull()]
print(len(open1),len(closed))
# score(caov2, car_type)
# print(caov2['Days_31_Plus'])
train_set, test_set = preprocessing1(closed, open1, car_type)
# train_set, test_set = preprocessing2(caov2, car_type)
# print(train_set['Days_31_Plus'])
ypred,yproba=training(train_set,test_set)
# ypred,yproba=final_model1(test_set)
print(type(ypred),type(yproba))
# print((np.array(test_set['Outstanding_Balance_1Feb'])*yproba[:,1]).sum())
