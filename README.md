# Loan Default Prediction

The loan default data was downloaded from kaggle. We have modelled the default prediction on given housing data. 

The following techniques have been used: 

 - Logistic regression
 - Decision Tree
 - Random Forest
 - XG Boost

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is F1-score.

## To excute the script
python <scriptname.py>

## To install the package
Go to the folder where wheel link was stored in the command prompt

pip install customer_analysis-0.3-py3-none-any.whl

## Scripts to check the installation
After successful installation, Go to python in command prompt
In python, type as from src.customer_analysis import chargeoff_probability
The log files are stored in the folder in which you are working.
