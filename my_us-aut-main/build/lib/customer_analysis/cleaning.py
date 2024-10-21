import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from .common import log
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings("ignore")

logging = log()


def spliting(us_auto):
    """_summary_

    :param us_auto: A pandas data frame is passed which consists of us_auto records with open and closed
    :type us_auto: pd.DataFrame
    :return: Returns train_set,test_set
    :rtype: Both train_set and test_set consists of pd.DataFrame
    """
    logging.info("Splitting of the dataset started")
    us_auto = (
        us_auto[us_auto["Account_Status"] == "CLOSED"]
        .reset_index()
        .drop("index", axis=1)
    )

    d = {"ChargeOff": 1, "PayOff": 0}
    us_auto["new_account_status"] = us_auto["Account_Status_New"].apply(
        lambda x: d[x]
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index, test_index in split.split(
        us_auto, us_auto["new_account_status"]
    ):
        train_set = us_auto.loc[train_index]
        test_set = us_auto.loc[test_index]
    train_set = train_set.reset_index().drop("index", axis=1)
    test_set = test_set.reset_index().drop("index", axis=1)
    logging.info("splitting completed")
    return train_set, test_set


def cleaning(train_set, test_set, ct):

    """_summary_

    :param train_set: It is a training data set
    :type train_set: pd.DataFrame
    :param test_set: It is a test data set
    :type test_set: pd.DataFrame
    :param ct: It is car type data set useful for grouping car models
    :type ct: pd.DataFrame
    :return: Returns train_set,test_set data sets
    :rtype: Both train_set and test_set consists of pd.DataFrame
    """
    logging.info("cleaning of the dataset started")
    train_set = train_set.reset_index().drop("index", axis=1)
    test_set = test_set.reset_index().drop("index", axis=1)
    
    d = {"ChargeOff": 1, "PayOff": 0}
    train_set["new_account_status"] = train_set["Account_Status_New"].apply(
        lambda x: d[x]
    )

    def gp(x):
        if (x == 59.00) | (x == 559.00) | (x == 599.99) | (x == 595.0):
            return 599.00
        if (x == 299.00) | (x == 399.99):
            return 399.00
        if x == 799.99:
            return 799.00
        if x == 999.0:
            return 699.0
        return x

    train_set["GAP_Price"] = train_set["GAP_Price"].apply(gp)
    test_set["GAP_Price"] = test_set["GAP_Price"].apply(gp)

    def vt(x):
        if x == 23.0 or x == 24000 or x == 240.0 or x == 26.0:
            return 24.0
        if x == 62 or x == 120.0:
            return 12
        if x == 0 or x == 36 or x == 12 or x == 24:
            return x
        return 36

    train_set["VSC_Term"] = train_set["VSC_Term"].apply(vt)
    test_set["VSC_Term"] = test_set["VSC_Term"].apply(vt)

    train_set["Added_Expense"] = train_set["Added_Expense"].apply(
        lambda x: np.abs(x)
    )
    test_set["Added_Expense"] = test_set["Added_Expense"].apply(
        lambda x: np.abs(x)
    )

    train_set["Make"] = train_set["Make"].apply(lambda x: x.upper())
    train_set["Make"] = train_set["Make"].apply(lambda x: x.strip())
    train_set["Make"] = train_set["Make"].replace("NIUSSAN", "NISSAN")
    train_set["Make"] = train_set["Make"].replace("GMC SIERRA", "GMC")
    train_set["Make"] = train_set["Make"].replace("CADILAC", "CADILLAC")
    train_set["Make"] = train_set["Make"].replace("KEEP", "JEEP")
    train_set["Make"] = train_set["Make"].replace("MISUBISHI", "MITSUBISHI")
    train_set["Make"] = train_set["Make"].replace("MERCEDES", "MERCEDES-BENZ")
    train_set["Make"] = train_set["Make"].replace("VOLKSWAGON", "VOLKSWAGEN")

    test_set["Make"] = test_set["Make"].apply(lambda x: x.upper())
    test_set["Make"] = test_set["Make"].apply(lambda x: x.strip())
    test_set["Make"] = test_set["Make"].replace("NIUSSAN", "NISSAN")
    test_set["Make"] = test_set["Make"].replace("GMC SIERRA", "GMC")
    test_set["Make"] = test_set["Make"].replace("CADILAC", "CADILLAC")
    test_set["Make"] = test_set["Make"].replace("KEEP", "JEEP")
    test_set["Make"] = test_set["Make"].replace("MISUBISHI", "MITSUBISHI")
    test_set["Make"] = test_set["Make"].replace("MERCEDES", "MERCEDES-BENZ")
    test_set["Make"] = test_set["Make"].replace("VOLKSWAGON", "VOLKSWAGEN")

    train_set["Customer_Age"] = np.clip(train_set["Customer_Age"], 19, 90)
    test_set["Customer_Age"] = np.clip(test_set["Customer_Age"], 19, 90)

    def fs(x):
        if x < 300:
            return 300
        return x

    train_set["FICO"] = train_set["FICO"].apply(fs)
    test_set["FICO"] = test_set["FICO"].apply(fs)

    train_set["Model_Score"] = train_set["Model_Score"].apply(fs)
    test_set["Model_Score"] = test_set["Model_Score"].apply(fs)

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

    train_set["Sale_Date"] = pd.to_datetime(train_set["Sale_Date"])
    test_set["Sale_Date"] = pd.to_datetime(test_set["Sale_Date"])

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
    train_set["Lot"] = train_set["Lot"].apply(lambda x: int(x.split("-")[0]))
    test_set["Lot"] = test_set["Lot"].apply(lambda x: int(x.split("-")[0]))

    def cs(x):
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

    train_set["Customer_State"] = train_set["Customer_State"].apply(cs)
    test_set["Customer_State"] = test_set["Customer_State"].apply(cs)

    train_set["Model"] = train_set["Model"].apply(
        lambda x: str(x).strip().upper()
    )
    test_set["Model"] = test_set["Model"].apply(
        lambda x: str(x).strip().upper()
    )
    train_set["Model"] = train_set["Model"].replace(
        "TOWN AND COUNTRY", "TOWN & COUNTRY"
    )
    train_set["Model"] = train_set["Model"].replace(
        "TOWN  AND COUNTRY", "TOWN & COUNTRY"
    )
    test_set["Model"] = test_set["Model"].replace(
        "TOWN AND COUNTRY", "TOWN & COUNTRY"
    )
    test_set["Model"] = test_set["Model"].replace(
        "TOWN  AND COUNTRY", "TOWN & COUNTRY"
    )

    ct["CARMAKE"] = ct["CARMAKE"].apply(lambda x: str(x).strip().upper())
    ct["CARMODEL"] = ct["CARMODEL"].apply(lambda x: str(x).strip().upper())
    d = ct.set_index(["CARMAKE", "CARMODEL"]).to_dict()["SEGMENTTYPE"]

    def ct1(x):
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

    train_set["Car_Type"] = train_set[["Make", "Model"]].apply(ct1, axis=1)
    test_set["Car_Type"] = test_set[["Make", "Model"]].apply(ct1, axis=1)

    def ct2(x):
        if pd.isnull(x[1]):
            if x[0] == "SUZUKI":
                return "SMALL SUV"
            return train_set[train_set["Make"] == x[0]]["Car_Type"].mode()[0]
        return x[1]

    train_set["Car_Type"] = train_set[["Make", "Car_Type"]].apply(ct2, axis=1)
    test_set["Car_Type"] = test_set[["Make", "Car_Type"]].apply(ct2, axis=1)

    # train_set['Days_7_plus']=train_set['Days_7_plus']-train_set['Days_14_plus']
    # train_set['Days_14_plus']=train_set['Days_14_plus']-train_set['Days_31_plus']
    # test_set['Days_7_plus']=test_set['Days_7_plus']-test_set['Days_14_plus']
    # test_set['Days_14_plus']=test_set['Days_14_plus']-test_set['Days_31_plus']

    train_set["Days_1_TO_15"] = (
        train_set["Days_1_TO_7"] + train_set["Days_8_TO_15"]
    )
    train_set["Days_31_Plus"] = (
        train_set["Days_31_TO_45"]
        + train_set["Days_46_TO_60"]
        + train_set["Days_61_TO_75"]
    )
    train_set["Days_31_Plus"] = (
        train_set["Days_31_Plus"]
        + train_set["Days_76_TO_90"]
        + train_set["Days_91_Plus"]
    )
    test_set["Days_1_TO_15"] = (
        test_set["Days_1_TO_7"] + test_set["Days_8_TO_15"]
    )
    test_set["Days_31_Plus"] = (
        test_set["Days_31_TO_45"]
        + test_set["Days_46_TO_60"]
        + test_set["Days_61_TO_75"]
    )
    test_set["Days_31_Plus"] = (
        test_set["Days_31_Plus"]
        + test_set["Days_76_TO_90"]
        + test_set["Days_91_Plus"]
    )

    logging.info("cleaning was completed")

    return train_set, test_set


def outlier_treatment(train_set, test_set):

    """_summary_

    :param train_set: It is a training data set
    :type train_set: pd.DataFrame
    :param test_set: It is a test data set
    :type test_set: pd.DataFrame
    :return: Returns train_set,test_set data sets
    :rtype: Data type of both train_set,test_set is pd.DataFrame
    """
    logging.info("outlier treatment was started")
    outlier = [
        "Income",
        "Cash_Down_Percentage",
        "Pick_Percentage",
        "Total_Down_Percentage",
        "Housing_Months",
        "Interest_Rate",
        "Original_Term",
        "Monthly_Eqv_Payment",
        "Sold_Price",
        "Car_Cost",
        "Added_Expense",
        "Customer_Age",
        "Model_Score",
        "Working_Months",
        "FICO",
        "Outbound_Call",
        "Inbound_Call",
        "Number_of_Payments",
        "Days_Current",
        "Original_Miles",
    ]
    for i in outlier:
        l1 = list(train_set[i].quantile([0.25, 0.5, 0.75]).values)
        upqu = (l1[2] - l1[0]) * 3 + l1[2]
        loqu = l1[0] - (l1[2] - l1[0]) * 3
        train_set[i] = np.clip(train_set[i], loqu, upqu)
        test_set[i] = np.clip(test_set[i], loqu, upqu)
    logging.info("outlier treatment was ended")
    return train_set, test_set


def encoding(train_set, test_set):
    """_summary_

    :param train_set: _description_
    :type train_set: _type_
    :param test_set: _description_
    :type test_set: _type_
    :return: _description_
    :rtype: _type_
    """

    logging.info("encoding was started")

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
        ohe.fit_transform(train_set[["Origination_State"]]).toarray()[:, :5]
    )
    test_set[["OS_AL", "OS_FL", "OS_GA", "OS_NC", "OS_S"]] = pd.DataFrame(
        ohe.transform(test_set[["Origination_State"]]).toarray()[:, 0:5]
    )

    ohe = OneHotEncoder()
    train_set[
        ["CS_AL", "CS_FL", "CS_GA", "CS_NC", "CS_SC", "CS_TN"]
    ] = pd.DataFrame(
        ohe.fit_transform(train_set[["Customer_State"]]).toarray()[:, 0:6]
    )
    test_set[
        ["CS_AL", "CS_FL", "CS_GA", "CS_NC", "CS_SC", "CS_TN"]
    ] = pd.DataFrame(
        ohe.transform(test_set[["Customer_State"]]).toarray()[:, 0:6]
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

    train_set.drop(
        [
            "Pay_Frequency",
            "Housing_Type",
            "VSC_Term",
            "GAP_Price",
            "Origination_State",
            "Customer_State",
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
            "Customer_State",
        ],
        axis=1,
        inplace=True,
    )

    d = {
        "COMPACT CAR": "4",
        "COUPE 2-DR": "5",
        "KICK": "4",
        "LARGE CAR": "2",
        "LARGE SUV": "2",
        "LARGE TRUCK": "2",
        "LUXURY LARGE CAR": "1",
        "LUXURY LARGE SUV": "1",
        "LUXURY MIDSIZE CAR": "1",
        "LUXURY MIDSIZE SEDAN": "1",
        "LUXURY MIDSIZE SUV": "1",
        "MEDIUM CAR": "3",
        "MEDIUM SUV": "3",
        "SMALL SUV": "4",
        "SMALL TRUCK": "4",
        "SPORTS CAR": "5",
        "VAN": "4",
    }
    train_set["Car_Type"] = train_set["Car_Type"].apply(lambda x: d[x])
    test_set["Car_Type"] = test_set["Car_Type"].apply(lambda x: d[x])
    ohe = OneHotEncoder()
    train_set[["CT_1", "CT_2", "CT_3", "CT_4"]] = pd.DataFrame(
        ohe.fit_transform(train_set[["Car_Type"]]).toarray()[:, :4]
    )
    test_set[["CT_1", "CT_2", "CT_3", "CT_4"]] = pd.DataFrame(
        ohe.transform(test_set[["Car_Type"]]).toarray()[:, :4]
    )

    logging.info("encoding was ended")
    return train_set, test_set


def imputation(train_set, test_set):

    """_summary_

    :param train_set: It is a training data set
    :type train_set: pd.DataFrame
    :param test_set: It is a test data set
    :type test_set: pd.DataFrame
    :return: Returns train_set,test_set data sets
    :rtype: Data type of both train_set,test_set is pd.DataFrame
    """

    logging.info("imputation was started")
    imp_med = [
        "Pick_Percentage",
        "Interest_Rate",
        "Customer_Age",
        "Housing_Months",
        "Working_Months",
        "Monthly_Eqv_Payment",
        "Model_Score",
    ]
    for i in imp_med:
        me = train_set[i].median()
        train_set[i] = train_set[i].fillna(me)
        test_set[i] = test_set[i].fillna(me)
    imp_mod = ["Housing_Type", "Pay_Frequency", "AppType", "Customer_Grade"]
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
        ["Sale_Date", "Original_Maturity_Date", "Original_Term"]
    ].apply(rd, axis=1)
    test_set["Original_Term"] = test_set[
        ["Sale_Date", "Original_Maturity_Date", "Original_Term"]
    ].apply(rd, axis=1)

    mo = train_set[train_set["GAP_Product"] == "Yes"]["GAP_Price"].mode()[0]
    train_set["GAP_Price"] = train_set["GAP_Price"].fillna(mo)
    test_set["GAP_Price"] = test_set["GAP_Price"].fillna(mo)

    me = train_set[train_set["Trade_Up_Flag"] != 0]["Trade_Amt"].median()

    def ta1(x):
        if pd.isnull(x[0]):
            if x[1] == 0:
                return 0
            else:
                return me
        return x[0]

    train_set["Trade_Amt"] = train_set[["Trade_Amt", "Trade_Up_Flag"]].apply(
        ta1, axis=1
    )
    test_set["Trade_Amt"] = test_set[["Trade_Amt", "Trade_Up_Flag"]].apply(
        ta1, axis=1
    )

    mo = train_set[train_set["VSC_Product"] != 0]["VSC_Term"].mode()[0]

    def vt(x):
        if pd.isnull(x[1]):
            if x[0] == "No":
                return 0
            else:
                return mo
        return x[1]

    train_set["VSC_Term"] = train_set[["VSC_Product", "VSC_Term"]].apply(
        vt, axis=1
    )
    test_set["VSC_Term"] = test_set[["VSC_Product", "VSC_Term"]].apply(
        vt, axis=1
    )

    def os(x):
        if pd.isnull(x[1]):
            return (
                train_set[train_set["Lot"] == x[0]]["Origination_State"]
                .value_counts()
                .index[0]
            )
        return x[1]

    train_set["Origination_State"] = train_set[
        ["Lot", "Origination_State"]
    ].apply(os, axis=1)
    test_set["Origination_State"] = test_set[
        ["Lot", "Origination_State"]
    ].apply(os, axis=1)

    def tdp(x):
        if pd.isnull(x[2]):
            return x[0] + x[1]
        return x[2]

    train_set["Total_Down_Percentage"] = train_set[
        ["Cash_Down_Percentage", "Pick_Percentage", "Total_Down_Percentage"]
    ].apply(tdp, axis=1)
    test_set["Total_Down_Percentage"] = test_set[
        ["Cash_Down_Percentage", "Pick_Percentage", "Total_Down_Percentage"]
    ].apply(tdp, axis=1)

    train_set.fillna(0, inplace=True)
    test_set.fillna(0, inplace=True)
    logging.info("imputation was ended")

    return train_set, test_set


def feature_selection_basic(train_set, test_set):

    """_summary_

    :param train_set: It is a training data set
    :type train_set: pd.DataFrame
    :param test_set: It is a test data set
    :type test_set: pd.DataFrame
    :return: Returns train_set,test_set data sets
    :rtype: Data type of both train_set,test_set is pd.DataFrame
    """

    train_set.drop(
        [
            "Unnamed: 0",
            "Pricing_Profile",
            "Account_Status",
            "Term_In_Months",
            "Lot",
            "Make",
            "APR",
            "Trade_Up_Flag",
            "Scoring_System",
            "System",
            "ChargeOff_Account",
            "Model",
            "Car_Type",
            "ChargeOff_Amount",
            "Net_Recovery",
            "Aggregate_Unpaid_Balance",
            "ZIP",
            "Primary_Grade",
            "Customer_Grade",
            "Customer_DOB",
            "Model_Type",
            "First_Name",
            "Last_Name",
            "PayOff_Date",
            "ChargeOff_Date",
            "Sold_Flag",
            "ChargeOff_Principal_Amount",
            "ChargeOff_Interest_Amount",
            "ChargeOff_Fee_Amount",
            "Net_Auction_Proceeds",
            "Total_Credits",
            "Payment",
            "Vehicle_VIN",
            "Primary_Score",
            "GL_Unit",
            "Account_Status_New",
            "Loan_Account",
            "Stock_Number",
            "Joint_Grade",
            "Joint_Score",
            "Customer_City",
            "IS_IN_BANKRUPTCY",
            "LAST_MONETARY_TRANS_TYPE",
            "LAST_PAYMENT_TRANS_TYPE",
            "LAST_MAINTENANCE_TRANS_TYPE",
            "LAST_SATISFIED_PAYMENT_DATE",
            "Cash_Down",
            "Pick_Payment",
            "Total_Down",
            "Months_Deferred",
            "Pmt_Deferrals",
            "Sale_Date",
            "Original_Maturity_Date",
        ],
        axis=1,
        inplace=True,
    )
    test_set.drop(
        [
            "Unnamed: 0",
            "Pricing_Profile",
            "Account_Status",
            "Term_In_Months",
            "Lot",
            "Make",
            "APR",
            "Trade_Up_Flag",
            "Scoring_System",
            "System",
            "ChargeOff_Account",
            "Model",
            "Car_Type",
            "ChargeOff_Amount",
            "Net_Recovery",
            "Aggregate_Unpaid_Balance",
            "ZIP",
            "Primary_Grade",
            "Customer_Grade",
            "Customer_DOB",
            "Model_Type",
            "First_Name",
            "Last_Name",
            "PayOff_Date",
            "ChargeOff_Date",
            "Sold_Flag",
            "ChargeOff_Principal_Amount",
            "ChargeOff_Interest_Amount",
            "ChargeOff_Fee_Amount",
            "Net_Auction_Proceeds",
            "Total_Credits",
            "Payment",
            "Vehicle_VIN",
            "Primary_Score",
            "GL_Unit",
            "Account_Status_New",
            "Loan_Account",
            "Stock_Number",
            "Joint_Grade",
            "Joint_Score",
            "Customer_City",
            "IS_IN_BANKRUPTCY",
            "LAST_MONETARY_TRANS_TYPE",
            "LAST_PAYMENT_TRANS_TYPE",
            "LAST_MAINTENANCE_TRANS_TYPE",
            "LAST_SATISFIED_PAYMENT_DATE",
            "Cash_Down",
            "Pick_Payment",
            "Total_Down",
            "Months_Deferred",
            "Pmt_Deferrals",
            "Sale_Date",
            "Original_Maturity_Date",
        ],
        axis=1,
        inplace=True,
    )
    return train_set, test_set


def additional_features(train_set, test_set):
    """_summary_

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

    train_set["PTI"] = train_set["Monthly_Eqv_Payment"] / train_set["Income"]
    test_set["PTI"] = test_set["Monthly_Eqv_Payment"] / test_set["Income"]

    train_set["Sale_Year"] = train_set["Sale_Date"].apply(lambda x: x.year)
    test_set["Sale_Year"] = test_set["Sale_Date"].apply(lambda x: x.year)

    train_set["Sale_Month"] = train_set["Sale_Date"].apply(lambda x: x.month)
    test_set["Sale_Month"] = test_set["Sale_Date"].apply(lambda x: x.month)

    return train_set, test_set
