"""It consists of a class, which performs different operations such as cleaning,imputation,encoding,outlier analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from .common import logg
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import StratifiedShuffleSplit
from .constants_for_files import (
    feature_selection,
    imp_med,
    imp_mod,
    outlier,
    car_tiers,
)


class DataProcessing:
    def __init__(self):
        self.logging = logg()

    def feature_selection_basic(
        self, train_set: pd.DataFrame, test_set: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):

        """It will take train_set,test_set pandas dataframes and drops unnecessary columns

        :param train_set: It is a training data set
        :type train_set: pd.DataFrame
        :param test_set: It is a test data set
        :type test_set: pd.DataFrame
        :return: Returns train_set,test_set data sets
        :rtype: Data type of both train_set,test_set is pd.DataFrame
        """

        train_set.drop(feature_selection, axis=1, inplace=True)
        test_set.drop(feature_selection, axis=1, inplace=True)
        return train_set, test_set

    def spliting(self, us_auto: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """It will take a single data frame and separate the closed ones among it. And it will perform stratified shuffle split

        :param us_auto: A pandas data frame is passed which consists of us_auto records with open and closed
        :type us_auto: pd.DataFrame
        :return: Returns train_set,test_set
        :rtype: Both train_set and test_set consists of pd.DataFrame
        """
        
        self.logging.info("Splitting of the dataset started")
        
        d={'OPEN':0,'CHARGEDOFF':1,'SETTLEMENT':1,'CUSTPAYOFF':0,'LPAYOFF':0}
        
        def nqas(x):
            if pd.isnull(x):
                return 4
            else:
                return d[x]
        us_auto['Next_Qtr_Account_Status']=us_auto['Next_Qtr_Account_Status'].apply(nqas)
        
        us_auto=us_auto[us_auto['Next_Qtr_Account_Status']!=4]
        
        us_auto=us_auto.reset_index().drop("index",axis=1)
                
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
        for train_index, test_index in split.split(
            us_auto, us_auto["Next_Qtr_Account_Status"]
        ):
            train_set = us_auto.loc[train_index]
            test_set = us_auto.loc[test_index]
#         print(len(us_auto[us_auto["Next_Qtr_Account_Status"]==1]),len(train_set[train_set["Next_Qtr_Account_Status"]==1]))
#         print(len(test_set[test_set["Next_Qtr_Account_Status"]==1]))
        train_set = train_set.reset_index().drop("index", axis=1)
        test_set = test_set.reset_index().drop("index", axis=1)
        self.logging.info("splitting completed")
        return train_set, test_set

    @staticmethod
    def gp(x: float) -> float:
        """This function will clean the outliers present in gap_price feature

        :param x: This is the gap_price
        :type x: float
        :return: Returns the correct value of gap_price
        :rtype: float
        """
        if (x == 59.00) | (x == 559.00) | (x == 599.99) | (x == 595.0):
            return 599.00
        if (x == 299.00) | (x == 399.99):
            return 399.00
        if x == 799.99:
            return 799.00
        if x == 999.0:
            return 699.0
        return x

    @staticmethod
    def vt(x: float) -> float:
        """It will clear the outliers present in vsc_term

        :param x: It is value of vsc_term which may contain outliers
        :type x: float
        :return: returns the correct value of vsc_term
        :rtype: float
        """
        if x == 23.0 or x == 24000 or x == 240.0 or x == 26.0:
            return 24.0
        if x == 62 or x == 120.0:
            return 12
        if x == 0 or x == 36 or x == 12 or x == 24:
            return x
        return 36

    @staticmethod
    def ct1(df: pd.DataFrame, d: dict) -> pd.DataFrame:
        """This will group the car models into tiers

        :param df: This is the dataframe
        :type df: pd.DataFrame
        :param d: it is the dictionary where values are car_tier
        :type d: dict
        :return: creates a new column and add it to the data frame and return that one
        :rtype: pd.DataFrame
        """

        def car_type(x: tuple) -> str:
            s = (x[0], x[1])
            if s in d.keys():
                return d[s]
            else:
                sp = x[1].split()
                s1 = sp[0]
                s = (x[0], s1)
                if s in d.keys():
                    return d[s]
                else:
                    for i in sp[1:]:
                        s1 = s1 + " " + i
                        s = (x[0], s1)
                        if s in d.keys():
                            return d[s]

        df["Car_Type"] = df[["Make", "Model"]].apply(car_type, axis=1)
        return df

    @staticmethod
    def ct2(df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        :param df: _description_
        :type df: pd.DataFrame
        :return: _description_
        :rtype: pd.DataFrame
        """

        def car_type(x: tuple) -> str:
            if pd.isnull(x[1]):
                if x[0] == "SUZUKI":
                    return "SMALL SUV"
                return df[df["Make"] == x[0]]["Car_Type"].mode()[0]
            return x[1]

        df["Car_Type"] = df[["Make", "Car_Type"]].apply(car_type, axis=1)
        return df

    @staticmethod
    def days_differ(df: pd.DataFrame, open_date: str) -> pd.DataFrame:
        """this method calculates the difference between sale date and charge_off or pay_off date for closed accounts. If the account is open then it will calculate the difference between sale_date and the date_entered

        :param x: It is a dataframe for which we need to calculate the difference between dates
        :type x: pd.DataFrame
        :param open_date: It is the date of when the data is fetched
        :type open_date: str
        :return: returns the dataframe after calculating the difference between two dates and and the result is stored in days_differ column
        :rtype: pd.DataFrame
        """

        def dd(x):
            if x[1] == "CLOSED":
                if pd.isnull(x[2]) and pd.isnull(x[3]):
                    return np.nan
                elif pd.isnull(x[2]):
                    return (x[3] - x[0]).days
                else:
                    return (x[2] - x[0]).days
            else:
                str_d1 = open_date
                d1 = datetime.strptime(str_d1, "%Y/%m/%d")
                return (d1 - x[0]).days

        df["days_differ"] = df[
            ["Sale_Date", "Account_Status", "PayOff_Date", "ChargeOff_Date"]
        ].apply(dd, axis=1)
        return df

    @staticmethod
    def cs(x: str) -> str:
        if (
            x == "GA"
            or x == "FL"
            or x == "SC"
            or x == "NC"
            or x == "TN"
            or x == "AL"
        ):
            return x
        return "VA"
    
    @staticmethod
    def new_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """This function takes a dataframe and create new column new_ratio

        :param df: It is a dataframe(takes both train_set and test_set)
        :type df: pd.DataFrame
        :return: It will return the dataframe after creating a new column in both train_set and test_set
        :rtype: pd.DataFrame
        """
        
        def new_ratio1(x: tuple) -> float:
            if x[0]==0 and x[1]==0:
                return 0
            if x[0]==0 and x[1]!=0:
                return 0
            if x[0]!=0 and x[1]==0:
                if x[0]<=15:
                    return 0.17
                elif x[0]<=40:
                    return 0.823
                else:
                    return 1.83
            if x[0]!=0 and x[1]!=0:
                return x[0]/x[1]

        df['new_ratio']=df[['Cum_Days_31_Plus','Cum_Days_16_TO_30']].apply(
            new_ratio1,axis=1
        )
        
        def ratio_cate(x: tuple) -> int:
            if x<=0.5:
                return 0
            if x<=1:
                return 1
            else:
                return 2
        df['new_ratio']=df['new_ratio'].apply(ratio_cate)
        return df


    def cleaning(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        ct: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        """It will take two data frames and performs different operations based on the respective column.

        :param train_set: It is a training data set
        :type train_set: pd.DataFrame
        :param test_set: It is a test data set
        :type test_set: pd.DataFrame
        :param ct: It is car type data set useful for grouping car models
        :type ct: pd.DataFrame
        :return: Returns train_set,test_set data sets
        :rtype: Both train_set and test_set consists of pd.DataFrame
        """
        
        d={'OPEN':0,'CHARGEDOFF':1,'SETTLEMENT':1,'CUSTPAYOFF':0,'LPAYOFF':0}
        
        def nqas(x):
            if pd.isnull(x):
                return 4
            else:
                if type(x)==str:
                    return d[x]
                else:
                    return x
        train_set['Next_Qtr_Account_Status']=train_set['Next_Qtr_Account_Status'].apply(nqas)
        self.logging.info("cleaning of the dataset started")
        train_set = train_set.reset_index().drop("index", axis=1)
        test_set = test_set.reset_index().drop("index", axis=1)
#         print(train_set["Next_Qtr_Account_Status"].value_counts())
        train_set["GAP_Price"] = train_set["GAP_Price"].apply(
            DataProcessing.gp
        )
        test_set["GAP_Price"] = test_set["GAP_Price"].apply(DataProcessing.gp)

        train_set["VSC_Term"] = train_set["VSC_Term"].apply(DataProcessing.vt)
        test_set["VSC_Term"] = test_set["VSC_Term"].apply(DataProcessing.vt)

        train_set["Added_Expense"] = train_set["Added_Expense"].apply(
            lambda x: np.abs(x)
        )
        test_set["Added_Expense"] = test_set["Added_Expense"].apply(
            lambda x: np.abs(x)
        )

        train_set["Make"] = (
            train_set["Make"]
            .apply(lambda x: x.upper().strip())
            .replace(
                {
                    "NIUSSAN": "NISSAN",
                    "GMC SIERRA": "GMC",
                    "CADILAC": "CADILLAC",
                    "KEEP": "JEEP",
                    "MISUBISHI": "MITSUBISHI",
                    "MERCEDES": "MERCEDES-BENZ",
                    "VOLKSWAGON": "VOLKSWAGEN",
                }
            )
        )

        test_set["Make"] = (
            test_set["Make"]
            .apply(lambda x: x.upper().strip())
            .replace(
                {
                    "NIUSSAN": "NISSAN",
                    "GMC SIERRA": "GMC",
                    "CADILAC": "CADILLAC",
                    "KEEP": "JEEP",
                    "MISUBISHI": "MITSUBISHI",
                    "MERCEDES": "MERCEDES-BENZ",
                    "VOLKSWAGON": "VOLKSWAGEN",
                }
            )
        )

        train_set["Customer_Age"] = np.clip(train_set["Customer_Age"], 19, 90)
        test_set["Customer_Age"] = np.clip(test_set["Customer_Age"], 19, 90)

        train_set["FICO"] = train_set["FICO"].apply(
            lambda x: 300 if x < 300 else x
        )
        test_set["FICO"] = test_set["FICO"].apply(
            lambda x: 300 if x < 300 else x
        )

        train_set["Model_Score"] = train_set["Model_Score"].apply(
            lambda x: 300 if x < 300 else x
        )
        test_set["Model_Score"] = test_set["Model_Score"].apply(
            lambda x: 300 if x < 300 else x
        )

        train_set["Customer_Grade"] = train_set["Customer_Grade"].apply(
            lambda x: "A" if x == "A+" else x
        )
        train_set["Customer_Grade"] = train_set["Customer_Grade"].apply(
            lambda x: "B" if x == "B+" else x
        )
        train_set["Customer_Grade"] = train_set["Customer_Grade"].apply(
            lambda x: "D" if x == "D+" else x
        )

        test_set["Customer_Grade"] = test_set["Customer_Grade"].apply(
            lambda x: "A" if x == "A+" else x
        )
        test_set["Customer_Grade"] = test_set["Customer_Grade"].apply(
            lambda x: "B" if x == "B+" else x
        )
        test_set["Customer_Grade"] = test_set["Customer_Grade"].apply(
            lambda x: "D" if x == "D+" else x
        )

        train_set = train_set[
            (train_set["Income"] > 500) & (train_set["Income"] < 10000)
        ]
        train_set = train_set.reset_index().drop("index", axis=1)

        train_set["SalesDate"] = pd.to_datetime(train_set["SalesDate"])
        test_set["SalesDate"] = pd.to_datetime(test_set["SalesDate"])

        train_set["Original_Maturity_Date"] = pd.to_datetime(
            train_set["Original_Maturity_Date"]
        )
        test_set["Original_Maturity_Date"] = pd.to_datetime(
            test_set["Original_Maturity_Date"]
        )

        def lo(x):
            a = str(x).replace(" -", "-")
            a = str(x).replace("- ", "-")
            return a

        train_set["Lot"] = train_set["Lot"].apply(lo)
        test_set["Lot"] = test_set["Lot"].apply(lo)
        train_set["Lot"] = train_set["Lot"].apply(
            lambda x: int(x.split("-")[0])
        )
        test_set["Lot"] = test_set["Lot"].apply(lambda x: int(x.split("-")[0]))

        train_set["CUSTOMER_STATE"] = train_set["CUSTOMER_STATE"].apply(
            DataProcessing.cs
        )
        test_set["CUSTOMER_STATE"] = test_set["CUSTOMER_STATE"].apply(
            DataProcessing.cs
        )

        train_set["Model"] = (
            train_set["Model"]
            .apply(lambda x: str(x).strip().upper())
            .replace(
                {
                    "TOWN AND COUNTRY": "TOWN & COUNTRY",
                    "TOWN  AND COUNTRY": "TOWN & COUNTRY",
                }
            )
        )

        test_set["Model"] = (
            test_set["Model"]
            .apply(lambda x: str(x).strip().upper())
            .replace(
                {
                    "TOWN AND COUNTRY": "TOWN & COUNTRY",
                    "TOWN  AND COUNTRY": "TOWN & COUNTRY",
                }
            )
        )

        ct["CARMAKE"] = ct["CARMAKE"].apply(lambda x: str(x).strip().upper())
        ct["CARMODEL"] = ct["CARMODEL"].apply(lambda x: str(x).strip().upper())
        d = ct.set_index(["CARMAKE", "CARMODEL"]).to_dict()["SEGMENTTYPE"]

        train_set = DataProcessing.ct1(train_set, d)
        test_set = DataProcessing.ct1(test_set, d)

        train_set = DataProcessing.ct2(train_set)
        test_set = DataProcessing.ct2(test_set)

        train_set["Cum_Days_1_TO_15"] = (
            train_set["Cum_Days_1_TO_7"] + train_set["Cum_Days_8_TO_15"]
        )
        train_set["Cum_Days_31_Plus"] = (
            train_set["Cum_Days_31_TO_45"]
            + train_set["Cum_Days_46_TO_60"]
            + train_set["Cum_Days_61_TO_75"]
            + train_set["Cum_Days_76_TO_90"]
            + train_set["Cum_Days_91_Plus"]
        )
        test_set["Cum_Days_1_TO_15"] = (
            test_set["Cum_Days_1_TO_7"] + test_set["Cum_Days_8_TO_15"]
        )
        test_set["Cum_Days_31_Plus"] = (
            test_set["Cum_Days_31_TO_45"]
            + test_set["Cum_Days_46_TO_60"]
            + test_set["Cum_Days_61_TO_75"]
            + test_set["Cum_Days_76_TO_90"]
            + test_set["Cum_Days_91_Plus"]
        )
        train_set["SalesDate"] = pd.to_datetime(train_set["SalesDate"])
        test_set["SalesDate"] = pd.to_datetime(test_set["SalesDate"])
        
        train_set=DataProcessing.new_ratio(train_set)
        test_set=DataProcessing.new_ratio(test_set)
        
        #         train_set["days_differ"] = train_set[["Sale_Date", "Account_Status", "PayOff_Date", "ChargeOff_Date"]].apply(
        #             DataProcessing.days_differ, axis=1
        #         )
        #         test_set["days_differ"] = test_set[["Sale_Date", "Account_Status", "PayOff_Date", "ChargeOff_Date"]].apply(
        #             DataProcessing.days_differ, axis=1
        #         )
       
        self.logging.info("cleaning was completed")

        return train_set, test_set

    @staticmethod
    def trade_amt(df: pd.DataFrame, me: float) -> pd.DataFrame:
        """This function takes a dataframe and handles null values present in the trade_amt

        :param df: It is a pandas dataframe
        :type df: pd.DataFrame
        :param me: It is the median of trade_amt
        :type me: float
        :return: It will return a dataframe after imputing the null values present in trade_amt
        :rtype: pd.DataFrame
        """

        def ta(x: tuple) -> float:
            if pd.isnull(x[0]):
                if x[1] == 0:
                    return 0
                else:
                    return me
            return x[0]

        df["Trade_Amt"] = df[["Trade_Amt", "Trade_Up_Flag"]].apply(ta, axis=1)
        return df

    @staticmethod
    def vsc_term(df: pd.DataFrame, mo: float) -> pd.DataFrame:
        """This function takes a dataframe and handles null values present in the vsc_term

        :param df: It is a pandas dataframe
        :type df: pd.DataFrame
        :param mo: It is the mode of the vsc_term
        :type mo: float
        :return: It will return a dataframe after imputing the null values present in vsc_term
        :rtype: pd.DataFrame
        """

        def vt(x: tuple) -> float:
            if pd.isnull(x[1]):
                if x[0] == "No":
                    return 0
                else:
                    return mo
            return x[1]

        df["VSC_Term"] = df[["VSC_Product", "VSC_Term"]].apply(vt, axis=1)
        return df

    @staticmethod
    def origination_state(df: pd.DataFrame) -> pd.DataFrame:
        """This function takes a dataframe and handle null values present in the origination_state and return the dataframe

        :param df: It is a dataframe(takes both train_set and test_set)
        :type df: pd.DataFrame
        :return: It will return the dataframe after imputing the origination_state
        :rtype: pd.DataFrame
        """

        def os(x: tuple) -> str:
            if pd.isnull(x[1]):
                return (
                    df[df["Lot"] == x[0]]["Origination_State"]
                    .value_counts()
                    .index[0]
                )
            return x[1]

        df["Origination_State"] = df[["Lot", "Origination_State"]].apply(
            os, axis=1
        )
        return df

    def imputation(
        self, train_set: pd.DataFrame, test_set: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        """It will take two pandas data frames and handles null values present in each columns

        :param train_set: It is a training data set
        :type train_set: pd.DataFrame
        :param test_set: It is a test data set
        :type test_set: pd.DataFrame
        :return: Returns train_set,test_set data sets
        :rtype: Data type of both train_set,test_set is pd.DataFrame
        """

        self.logging.info("imputation was started")
        for i in imp_med:
            me = train_set[i].median()
            train_set[i] = train_set[i].fillna(me)
            test_set[i] = test_set[i].fillna(me)

        for i in imp_mod:
            mo = train_set[i].mode()[0]
            train_set[i] = train_set[i].fillna(mo)
            test_set[i] = test_set[i].fillna(mo)

        def rd(x):
            if pd.isnull(x[2]):
                a = relativedelta(x[1], x[0])
                return a.months + ((a.years) * 12) - 1
            else:
                return x[2]

        train_set["Original_Term"] = train_set[
            ["SalesDate", "Original_Maturity_Date", "Original_Term"]
        ].apply(rd, axis=1)
        test_set["Original_Term"] = test_set[
            ["SalesDate", "Original_Maturity_Date", "Original_Term"]
        ].apply(rd, axis=1)

        mo = train_set[train_set["GAP_Product"] == "Yes"]["GAP_Price"].mode()[
            0
        ]
        train_set["GAP_Price"] = train_set["GAP_Price"].fillna(mo)
        test_set["GAP_Price"] = test_set["GAP_Price"].fillna(mo)

        me = train_set[train_set["Trade_Up_Flag"] != 0]["Trade_Amt"].median()
        train_set = DataProcessing.trade_amt(train_set, me)
        test_set = DataProcessing.trade_amt(test_set, me)

        mo = train_set[train_set["VSC_Product"] != 0]["VSC_Term"].mode()[0]
        train_set = DataProcessing.vsc_term(train_set, mo)
        test_set = DataProcessing.vsc_term(test_set, mo)

        train_set = DataProcessing.origination_state(train_set)
        test_set = DataProcessing.origination_state(test_set)

        def tdp(x):
            if pd.isnull(x[2]):
                return x[0] + x[1]
            return x[2]

        train_set["Total_Down_Percentage"] = train_set[
            [
                "Cash_Down_Percentage",
                "Pick_Percentage",
                "Total_Down_Percentage",
            ]
        ].apply(tdp, axis=1)
        test_set["Total_Down_Percentage"] = test_set[
            [
                "Cash_Down_Percentage",
                "Pick_Percentage",
                "Total_Down_Percentage",
            ]
        ].apply(tdp, axis=1)

        train_set.fillna(0, inplace=True)
        test_set.fillna(0, inplace=True)
        self.logging.info("imputation was ended")

        return train_set, test_set

    def outlier_treatment(
        self, train_set: pd.DataFrame, test_set: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        """It will take two pandas data frames and perform outlier analysis on the columns

        :param train_set: It is a training data set
        :type train_set: pd.DataFrame
        :param test_set: It is a test data set
        :type test_set: pd.DataFrame
        :return: Returns train_set,test_set data sets
        :rtype: Data type of both train_set,test_set is pd.DataFrame
        """
        self.logging.info("outlier treatment was started")

        for i in outlier:
            l1 = list(train_set[i].quantile([0.25, 0.5, 0.75]).values)
            upqu = (l1[2] - l1[0]) * 3 + l1[2]
            loqu = l1[0] - (l1[2] - l1[0]) * 3
            train_set[i] = np.clip(train_set[i], loqu, upqu)
            test_set[i] = np.clip(test_set[i], loqu, upqu)
        self.logging.info("outlier treatment was ended")
        return train_set, test_set

    def encoding(
        self, train_set: pd.DataFrame, test_set: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):

        """It takes two data frames and perform encoding on the categorical columns

        :param train_set: It is a training data set
        :type train_set: pd.DataFrame
        :param test_set: It is a test data set
        :type test_set: pd.DataFrame
        :return: Returns train_set,test_set data sets
        :rtype: Data type of both train_set,test_set is pd.DataFrame
        """

        self.logging.info("encoding was started")

        ohe = OneHotEncoder()

        train_set[["Pay_Frequency_B", "Pay_Frequency_M"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["Pay_Frequency"]]).toarray()[:, 0:2]
        )
        test_set[["Pay_Frequency_B", "Pay_Frequency_M"]] = pd.DataFrame(
            ohe.transform(test_set[["Pay_Frequency"]]).toarray()[:, 0:2]
        )

        train_set["GAP_Product"] = train_set["GAP_Product"].apply(
            lambda x: 0 if x == "No" else 1
        )
        test_set["GAP_Product"] = test_set["GAP_Product"].apply(
            lambda x: 0 if x == "No" else 1
        )

        train_set["VSC_Product"] = train_set["VSC_Product"].apply(
            lambda x: 0 if x == "No" else 1
        )
        test_set["VSC_Product"] = test_set["VSC_Product"].apply(
            lambda x: 0 if x == "No" else 1
        )

        train_set["AppType"] = train_set["AppType"].apply(
            lambda x: 0 if x == "SINGLE" else 1
        )
        test_set["AppType"] = test_set["AppType"].apply(
            lambda x: 0 if x == "SINGLE" else 1
        )

        ohe = OneHotEncoder()
        train_set[["Housing_Type_Other", "Housing_Type_Own"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["Housing_Type"]]).toarray()[:, 0:2]
        )
        test_set[["Housing_Type_Other", "Housing_Type_Own"]] = pd.DataFrame(
            ohe.transform(test_set[["Housing_Type"]]).toarray()[:, 0:2]
        )

        ohe = OneHotEncoder()
        train_set[["OS_AL", "OS_FL", "OS_GA", "OS_NC", "OS_S"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["Origination_State"]]).toarray()[
                :, :5
            ]
        )
        test_set[["OS_AL", "OS_FL", "OS_GA", "OS_NC", "OS_S"]] = pd.DataFrame(
            ohe.transform(test_set[["Origination_State"]]).toarray()[:, 0:5]
        )

        ohe = OneHotEncoder()
        train_set[
            ["CS_AL", "CS_FL", "CS_GA", "CS_NC", "CS_SC", "CS_TN"]
        ] = pd.DataFrame(
            ohe.fit_transform(train_set[["CUSTOMER_STATE"]]).toarray()[:, 0:6]
        )
        test_set[
            ["CS_AL", "CS_FL", "CS_GA", "CS_NC", "CS_SC", "CS_TN"]
        ] = pd.DataFrame(
            ohe.transform(test_set[["CUSTOMER_STATE"]]).toarray()[:, 0:6]
        )

        ohe = OneHotEncoder()
        train_set[["CG_A", "CG_B", "CG_C", "CG_C+"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["Customer_Grade"]]).toarray()[:, 0:4]
        )
        test_set[["CG_A", "CG_B", "CG_C", "CG_C+"]] = pd.DataFrame(
            ohe.transform(test_set[["Customer_Grade"]]).toarray()[:, 0:4]
        )

        ohe = OneHotEncoder()
        train_set[["GP_0", "GP_399", "GP_599", "GP_699"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["GAP_Price"]]).toarray()[:, :4]
        )
        test_set[["GP_0", "GP_399", "GP_599", "GP_699"]] = pd.DataFrame(
            ohe.transform(test_set[["GAP_Price"]]).toarray()[:, :4]
        )

        ohe = OneHotEncoder()
        train_set[["VT_0", "VT_12", "VT_24"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["VSC_Term"]]).toarray()[:, :3]
        )
        test_set[["VT_0", "VT_12", "VT_24"]] = pd.DataFrame(
            ohe.transform(test_set[["VSC_Term"]]).toarray()[:, :3]
        )
        
        ohe = OneHotEncoder(categories=[[0,1,2]])
        train_set[["nr_1", "nr_2"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["new_ratio"]]).toarray()[:, :2]
        )
        test_set[["nr_1", "nr_2"]] = pd.DataFrame(
            ohe.transform(test_set[["new_ratio"]]).toarray()[:, :2]
        )

        train_set.drop(
            [
                "Pay_Frequency",
                "Housing_Type",
                "VSC_Term",
                "GAP_Price",
                "Origination_State",
                "CUSTOMER_STATE",
            ],
            axis=1,
            inplace=True,
        )
        test_set.drop(
            [
                "Pay_Frequency",
                "Housing_Type",
                "VSC_Term",
                "GAP_Price",
                "Origination_State",
                "CUSTOMER_STATE",
            ],
            axis=1,
            inplace=True,
        )

        train_set["Car_Type"] = train_set["Car_Type"].apply(
            lambda x: car_tiers[x]
        )
        test_set["Car_Type"] = test_set["Car_Type"].apply(
            lambda x: car_tiers[x]
        )
        ohe = OneHotEncoder()
        train_set[["CT_1", "CT_2", "CT_3", "CT_4"]] = pd.DataFrame(
            ohe.fit_transform(train_set[["Car_Type"]]).toarray()[:, :4]
        )
        test_set[["CT_1", "CT_2", "CT_3", "CT_4"]] = pd.DataFrame(
            ohe.transform(test_set[["Car_Type"]]).toarray()[:, :4]
        )

        self.logging.info("encoding was ended")
        return train_set, test_set
    
    
    def additional_features(
        self, train_set: pd.DataFrame, test_set: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        """It will create new features based on the existing ones

        :param train_set: It is a training data set
        :type train_set: pd.DataFrame
        :param test_set: It is a test data set
        :type test_set: pd.DataFrame
        :return: returns train_set,test_set datasets
        :rtype: Data type of both train_set,test_set is pd.DataFrame
        """
        train_set["Total_Car_Cost"] = (
            train_set["Car_Cost"] + train_set["Added_Expense"]
        )
        test_set["Total_Car_Cost"] = (
            test_set["Car_Cost"] + test_set["Added_Expense"]
        )

        train_set["PTI"] = (
            train_set["Monthly_Eqv_Payment"] / train_set["Income"]
        )
        test_set["PTI"] = test_set["Monthly_Eqv_Payment"] / test_set["Income"]

        train_set["Sale_Year"] = train_set["SalesDate"].apply(lambda x: x.year)
        test_set["Sale_Year"] = test_set["SalesDate"].apply(lambda x: x.year)

        train_set["Sale_Month"] = train_set["SalesDate"].apply(
            lambda x: x.month
        )
        test_set["Sale_Month"] = test_set["SalesDate"].apply(lambda x: x.month)

        return train_set, test_set
