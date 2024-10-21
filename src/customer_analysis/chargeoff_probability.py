import os
import os.path as op
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
import pickle
import warnings
from pathlib import Path
from .data_preprocessing import DataProcessing

BASE_DIR = Path(".").resolve().parent

warnings.filterwarnings("ignore")


def training(train_set, test_set):
    """_summary_

    :param train_set: It is a train data set
    :type train_set: pd.DataFrame
    :param test_set: It is a test data set
    :type test_set: pd.DataFrame
    :return: return prediction of test_set and probability of charge off and payoff
    :rtype: yproba is np.array where ypred is pd.Seried
    """
    l1 = [
        "Cum_Days_31_Plus",
        "Cum_Days_16_TO_30",
        "Cum_Days_1_TO_15",
        "Cum_Payment_Transactions",
        "Cum_Late_Fee_Charged",
        "Cum_Outbound_Call_LM",
        "Cum_Inbound_Call",
        "GP_399",
        "Original_Miles",
        "Cum_Outbound_Call",
        "Housing_Months",
        "Working_Months",
        "Cum_Promises_Broken",
        "Cum_Late_Fee_Paid",
        "Cum_Promises_Kept",
        "Outstanding_Percentage",
        "Qtr",
        "nr_1",
        "nr_2",
        "Next_Qtr_Account_Status"
    ]
    z = [
        "Cum_Days_31_Plus",
        "Cum_Days_16_TO_30",
        "Cum_Days_1_TO_15",
        "Cum_Payment_Transactions",
        "Cum_Late_Fee_Charged",
        "Cum_Outbound_Call_LM",
        "Cum_Inbound_Call",
        "GP_399",
        "Original_Miles",
        "Cum_Outbound_Call",
        "Housing_Months",
        "Working_Months",
        "Cum_Promises_Broken",
        "Cum_Late_Fee_Paid",
        "Cum_Promises_Kept",
        "Outstanding_Percentage",
        "Qtr",
        "nr_1",
        "nr_2"
       ]
    train_set = train_set[l1]
    chargof=train_set[train_set['Next_Qtr_Account_Status']==1]
    l3=chargof['Qtr'].value_counts().sort_index().values
    ope1=pd.DataFrame(columns=chargof.columns)
    for i in range(len(l3)):
        ope1 = pd.concat(
            [
                train_set[
                    (train_set["Next_Qtr_Account_Status"] == 0)
                    & (train_set["Qtr"] == i)
                ]
                .sample(l3[i], replace=True)
                .reset_index()
                .drop("index", axis=1),
                ope1,
                ]
            ) 
    train_set=pd.concat([ope1,chargof]).reset_index().drop('index',axis=1)
    test_set = test_set[z]
    rfr = RandomForestClassifier(min_samples_split=10, max_features="auto")
    xtrain=train_set.drop('Next_Qtr_Account_Status',axis=1)
    ytrain=train_set['Next_Qtr_Account_Status']
    xtest=test_set
#     xtest=test_set.drop('Next_Qtr_Account_Status',axis=1)
#     ytest=test_set['Next_Qtr_Account_Status']
    ytrain=ytrain.astype('int')
    for i in xtrain.columns:
        xtrain[i]=xtrain[i].astype('float')
    rfr.fit(xtrain,ytrain)
    ypred = rfr.predict(xtest)
    yproba = rfr.predict_proba(xtest)
#     print(classification_report(ytest,ypred))
#     print(confusion_matrix(ytest,ypred))
#     print(roc_auc_score(ytest,ypred))
    h = op.dirname(op.abspath(__file__))
    csv_path3 = os.path.join(h, "final_model.pkl")
    l1 = csv_path3.split("\\")

    l2 = "\\\\".join(l1)
    pickle.dump(rfr, open(l2, "wb"))
    return ypred, yproba


def final_model1(test_set):
    """_summary_

    :param train_set: It is a train data set
    :type train_set: pd.DataFrame
    :param test_set: It is a test data set
    :type test_set: pd.DataFrame
    :return: return prediction of test_set and probability of charge off and payoff
    :rtype: yproba is np.array where ypred is pd.Seried
    """

    z = [
        "Cum_Days_31_Plus",
        "Cum_Days_16_TO_30",
        "Cum_Days_1_TO_15",
        "Cum_Payment_Transactions",
        "Cum_Late_Fee_Charged",
        "Cum_Outbound_Call_LM",
        "Cum_Inbound_Call",
        "GP_399",
        "Original_Miles",
        "Cum_Outbound_Call",
        "Housing_Months",
        "Working_Months",
        "Cum_Promises_Broken",
        "Cum_Late_Fee_Paid",
        "Cum_Promises_Kept",
        "Outstanding_Percentage",
        "Qtr",
        "nr_1",
        "nr_2"
    ]
    h = op.dirname(op.abspath(__file__))
    csv_path3 = os.path.join(h, "final_model.pkl")
    l1 = csv_path3.split("\\")

    l2 = "\\\\".join(l1)
    l1 = l2 + "\\\\artifacts\\\\"
    
    #     print(h,l2,l,csv_path3,1)
    
#     print(BASE_DIR,l2)
    pa = BASE_DIR / "src" / "customer_analysis" / "final_model.pkl"
#     print(pa)
    rfr = pickle.load(open(l2, "rb"))
    test_set = test_set[z]
    ypred = rfr.predict(test_set)
    yproba = rfr.predict_proba(test_set)
#     print(ypred, yproba)
    return ypred, yproba


def score(train_set, car_type):
    """_summary_

    :param train_set: It is a train data set
    :type train_set: pd.DataFrame
    :param car_type: It is a test data set
    :type car_type: pd.DataFrame
    """
    dp=DataProcessing()
    train_set, test_set = dp.spliting(train_set)
    train_set1, test_set1 = dp.cleaning(train_set, test_set, car_type)
    train_set1, test_set1 = dp.imputation(train_set1, test_set1)
    train_set1, test_set1 = dp.outlier_treatment(train_set1, test_set1)
    train_set1, test_set1 = dp.encoding(train_set1, test_set1)
    train_set1, test_set1 = dp.additional_features(train_set1, test_set1)
#     train_set1, test_set1 = dp.feature_selection_basic(train_set1, test_set1)
    mdf = (
        pd.concat([train_set1, test_set1]).reset_index().drop("index", axis=1)
    )
    kf = KFold(n_splits=10)
    for train_set, test_set in kf.split(mdf.index):
        train_set_df = mdf.loc[train_set]
        test_set_df = mdf.loc[test_set]
        train_set_df = train_set_df.reset_index().drop("index", axis=1)
        test_set_df = test_set_df.reset_index().drop("index", axis=1)
        rfr = RandomForestClassifier()
        l1 = [
            "Cum_Days_31_Plus",
            "Cum_Days_16_TO_30",
            "Cum_Days_1_TO_15",
            "Cum_Payment_Transactions",
            "Cum_Late_Fee_Charged",
            "Cum_Outbound_Call_LM",
            "Cum_Inbound_Call",
            "GP_399",
            "Original_Miles",
            "Cum_Outbound_Call",
            "Housing_Months",
            "Working_Months",
            "Cum_Promises_Broken",
            "Cum_Late_Fee_Paid",
            "Cum_Promises_Kept",
            "Outstanding_Percentage",
            "Qtr",
            "nr_1",
            "nr_2",
            "Next_Qtr_Account_Status"
        ]
        train_set_df = train_set_df[l1]
        chargof=train_set_df[train_set_df['Next_Qtr_Account_Status']==1]
        l3=chargof['Qtr'].value_counts().sort_index().values
        ope1=pd.DataFrame(columns=chargof.columns)
        for i in range(len(l3)):
            ope1 = pd.concat(
                [
                    train_set_df[
                        (train_set_df["Next_Qtr_Account_Status"] == 0)
                        & (train_set_df["Qtr"] == i)
                    ]
                    .sample(l3[i], replace=True)
                    .reset_index()
                    .drop("index", axis=1),
                    ope1,
                ]
            ) 
        print(len(l3),len(chargof),len(ope1))
        train_set_df=pd.concat([ope1,chargof]).reset_index().drop('index',axis=1)
        print(len(train_set_df),len(train_set_df.columns))
        print(train_set_df["Next_Qtr_Account_Status"].value_counts())
        test_set_df = test_set_df[l1]
        xtrain = train_set_df.drop("Next_Qtr_Account_Status", axis=1)
        ytrain = train_set_df["Next_Qtr_Account_Status"]
        xtest = test_set_df.drop("Next_Qtr_Account_Status", axis=1)
        ytest = test_set_df["Next_Qtr_Account_Status"]
        ytrain=ytrain.astype('int')
        for i in xtrain.columns:
            xtrain[i]=xtrain[i].astype('float')
        #         print(xtrain.head())
        rfr.fit(xtrain, ytrain)
        ypred = rfr.predict(xtest)
        print(f"Accuracy_Score {accuracy_score(ytest,ypred)}")
        print(f"Roc_auc_score {roc_auc_score(ytest,ypred)}")
        print(classification_report(ytest,ypred))
        print(confusion_matrix(ytest,ypred))
