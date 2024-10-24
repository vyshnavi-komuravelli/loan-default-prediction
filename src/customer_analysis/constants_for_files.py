feature_selection=[
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
            ]

imp_med = [
            "Pick_Percentage",
            "Interest_Rate",
            "Customer_Age",
            "Housing_Months",
            "Working_Months",
            "Monthly_Eqv_Payment",
            "Model_Score",
        ]
imp_mod = ["Housing_Type", "Pay_Frequency", "AppType", "Customer_Grade"]

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
            "Cum_Days_Current",
            "Original_Miles",
        ]

car_tiers = {
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