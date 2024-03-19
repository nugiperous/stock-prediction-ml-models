# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:24:34 2022

@author: agarc
"""

import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# Setup
# Create Full Path - This is the OS agnostic way of doing so
dir_name = os.getcwd()
filename = 'GM_5Y_HistoricalData.csv'
full_path = os.path.join(dir_name, filename)

#
# Create the Main Data Frame
#
df_main = pd.read_csv(full_path) # read Excel spreadsheet
df_main.name = 'df_main' # name it
df_main['Date'] = pd.to_datetime(df_main['Date']) # Convert to Datetime


# - Create Data Frames for the other Features -
# COMP (NASDAQ) Index
NASDAQ_df = pd.read_csv(os.path.join(dir_name, 'COMP_5Y_HistoricalData.csv'))
NASDAQ_df['Date'] = pd.to_datetime(NASDAQ_df['Date']) # Convert to Datetime
NASDAQ_df = NASDAQ_df.drop(columns = ['Volume']) # Drop Volume
NASDAQ_df.name = "NASDAQ" # name it

# NYSE Index
NYSE_df = pd.read_csv(os.path.join(dir_name, 'NYA_5Y_HistoricalData.csv'))
NYSE_df['Date'] = pd.to_datetime(NYSE_df['Date'])
NYSE_df = NYSE_df.drop(columns = ['Volume'])
NYSE_df.name = 'NYSE'

#S&P 500
SP500_df = pd.read_csv(os.path.join(dir_name, 'SPX_5Y_HistoricalData.csv'))
SP500_df['Date'] = pd.to_datetime(SP500_df['Date'])
SP500_df = SP500_df.drop(columns = ['Volume'])
SP500_df.name = 'SP500'

# Oil
oil_df = pd.read_csv(os.path.join(dir_name, 'Oil_04_28_22-05_01_17.csv'))
oil_df['Date'] = pd.to_datetime(oil_df['Date'])
oil_df = oil_df.drop(columns = ['Volume'])
oil_df.name = 'oil'

# Natural Gas
natural_gas_df = pd.read_csv(os.path.join(dir_name, 'Natural_Gas_04_28_22-05_01_17.csv'))
natural_gas_df['Date'] = pd.to_datetime(natural_gas_df['Date'])
natural_gas_df = natural_gas_df.drop(columns = ['Volume'])
natural_gas_df.name = 'natural_gas'

# Commodities
commodity_df = pd.read_excel(os.path.join(dir_name, 'commodity-price-index-60.xlsx'))
commodity_df.name = 'commodity'
commodity_df['Month'] = pd.to_datetime(commodity_df['Month'])

# Inflation
inflation_df = pd.read_excel(os.path.join(dir_name, 'Inflation_rate_per_year.xlsx'))
inflation_df = inflation_df[inflation_df['Country Name'] == 'United States']
inflation_df = inflation_df.reset_index(drop = True)
inflation_df = inflation_df.drop(columns = ['Country Name','Country Code', 'Indicator Name', 'Indicator Code'])
inflation_df.name = 'inflation'


# Combine Stock Data to df_main
item_list = [NASDAQ_df, NYSE_df, SP500_df, oil_df, natural_gas_df]

for df in tqdm(item_list):
    for idx in df_main.index:
        this_series = df[df['Date'] == df_main['Date'][idx]]
        for this_column in this_series.columns:
            try:
                df_main.loc[idx,str(df.name)+'_'+str(this_column)] = this_series[this_column].values[0]
            except:
                df_main.loc[idx,str(df.name)+'_'+str(this_column)] = np.NAN

# Drop Dates from Features

for item in item_list:
    df_main = df_main.drop(columns = [item.name+'_Date'])


# Generating a Report for RAW
import sys
stats_path = os.path.join(dir_name,'..', 'Homework_2')
sys.path.append(stats_path)
from stats_report import StatsReport

labels = df_main.columns
report = StatsReport()

# Create a simple data set summary for the console
for thisLabel in tqdm(labels): # for each column, report stats
    thisCol = df_main[thisLabel]
    report.addCol(thisLabel, thisCol)

#print(report.to_string())
report.statsdf.to_excel("Quality_Report_Before_Prep.xlsx")


# Replace all Missing Values then Make Report - Should have no missing values
labels = df_main.columns
report = StatsReport()

for this_label in labels:
    df_main[this_label].fillna(df_main[this_label].mean(), inplace = True)

# Create a simple data set summary for the console
for thisLabel in tqdm(labels): # for each column, report stats
    thisCol = df_main[thisLabel]
    report.addCol(thisLabel, thisCol)

report.statsdf.to_excel("Quality_Report.xlsx")
 
# Add Several Day Data

labels = df_main.columns
for this_label in labels:
    print(f'\nUsing Label: {this_label}')
    for idx in tqdm(df_main.index):
        if idx < 3 or idx > len(df_main):
            pass
        else:
            df_main.loc[idx, this_label + '_in_1_day'] = df_main.loc[idx-1][this_label]
            df_main.loc[idx, this_label + '_in_2_days'] = df_main.loc[idx-2][this_label]
            df_main.loc[idx, this_label + '_in_3_days'] = df_main.loc[idx-3][this_label]
# Add Commodities and Inflation

for c_idx in commodity_df.index:
    for df_idx in tqdm(df_main.index):
        if commodity_df['Month'][c_idx].month == df_main['Date'][df_idx].month:
            #print(f'\nAdding Here: c_idx = {c_idx}, df_idx = {df_idx}')
            df_main.loc[df_idx,str(commodity_df.name)+'_'+str('Price')] = commodity_df['Price'][c_idx]
            df_main.loc[df_idx,str(commodity_df.name)+'_'+str('Change')] = commodity_df['Change'][c_idx]
        else:
            #print("Didn't Work")
            pass
        
for year in inflation_df.columns:
    for df_idx in tqdm(df_main.index):
        if year == df_main['Date'][df_idx].year:
            print('\nworked')
            df_main.loc[df_idx,'Inflation_Rate'] = inflation_df[year][0]

# Replace Inflation Rate of 2022 with Avg
df_main['Inflation_Rate'].fillna(df_main['Inflation_Rate'].mean(), inplace = True)

# Create Target Value - Stock Price 28 days later
import datetime

start = df_main['Date'][0] - datetime.timedelta(days = 28)
idx = df_main[df_main['Date'] == start].index[0]

for this_idx in tqdm(range(idx,len(df_main))):
    future_date = df_main['Date'] == df_main['Date'][this_idx] + datetime.timedelta(days = 28)
    if future_date.sum() == 0:
        df_main.loc[this_idx, 'GM_Close/Last_in_28_Days'] = df_main['GM_Close/Last'].mean()
    else:
        future_idx = df_main[future_date].index[0]
        df_main.loc[this_idx, 'GM_Close/Last_in_28_Days'] = df_main['GM_Close/Last'][future_idx]

df_main = df_main.drop(index = range(0,idx)).reset_index(drop = True)

#%%
# Assuming I buy GM stock every time model says yes to profit in 28 days
# Assuming over 3 Months period
def calculate_income(clf, X, df):
    predY = clf.predict(X)
    total_income = 0
    for current_price, pred_future_price, actual_future_price in zip(df['GM_Close/Last'], predY, df['GM_Close/Last_in_28_Days'] ):
        if current_price < pred_future_price:
            income = (1000/current_price)*actual_future_price - 1000
        else:
            income = 0
        total_income = total_income+income
    return total_income

#%% Imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#%% Target Data
# Setting up Training Data
# Data
predictors = df_main.drop(columns = ['GM_Close/Last_in_28_Days']).columns
X = df_main[predictors].to_numpy(np.float64)
min_max_scaler = MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)

Y = df_main['GM_Close/Last_in_28_Days'].to_numpy(np.float64)

# Split Testing and Training Data
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.3, 
                                                  train_size=0.7, random_state=22222, 
                                                  shuffle=True, stratify=None) 

#%%
# Linear Regression
linreg_model = LinearRegression().fit(X_train, y_train)
y_pred = linreg_model.predict(X_test)
print(f'Linear Regression Score: {linreg_model.score(X_test, y_test):.4f}')
print(f'Mean squared error (MSE): {mean_squared_error(y_test, y_pred):.4f}')
#print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

print(f"Income: {calculate_income(linreg_model, X_norm, df_main)}")

#%% Create Binary Target DataFrame
df_bin_target = df_main.copy()
for idx in df_bin_target.index:
    if df_bin_target.loc[idx, 'GM_Close/Last'] < df_bin_target.loc[idx, 'GM_Close/Last_in_28_Days']:
        df_bin_target.loc[idx, 'Profit'] = True
    else:
        df_bin_target.loc[idx, 'Profit'] = False
        
# Setting up Training Data
predictors = df_bin_target.drop(columns = ['Profit']).columns
X_bin = df_bin_target[predictors].to_numpy(np.float64)
min_max_scaler = MinMaxScaler()
X_bin_norm = min_max_scaler.fit_transform(X_bin)

Y_bin = df_bin_target['Profit'].to_numpy(np.float64)

# Split Testing and Training Data
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin_norm, Y_bin, test_size=0.3, 
                                                                    train_size=0.7, random_state=22222, 
                                                                    shuffle=True, stratify=None)
#%% Logistic Regression
modelLog = LogisticRegression()
clf_Log = modelLog.fit(X_bin_train, y_bin_train)
Ypred = clf_Log.predict(X_bin_test)
Ypredclass = 1*(Ypred > 0.5)
print("Logistic Regression \nR2 = %f,  MSE = %f,  Classification Accuracy = %f" % (metrics.r2_score(y_bin_test, Ypred), metrics.mean_squared_error(y_bin_test, Ypred), metrics.accuracy_score(y_bin_test, Ypredclass)))

Ypred = clf_Log.predict(X_bin_norm)
total_income = 0
for current_price, pred, actual_future_price in zip(df_main['GM_Close/Last'], Ypred, df_main['GM_Close/Last_in_28_Days'] ):
    if pred == 1:
        income = (1000/current_price)*actual_future_price - 1000
    else:
        income = 0
    total_income = total_income+income

print(f'Income: {total_income}')
#%% - MIGHT NOT NEED - DELETE LATER
'''
# Poly
poly = PolynomialFeatures(2) # object to generate polynomial basis functions
#X1_train = df_main.drop([targetName], axis=1).to_numpy()
bigTrainX = poly.fit_transform(X1_train)
mlrf = LogisticRegression() # creates the regressor object
mlrf.fit(bigTrainX, y_train)
Ypred1 = mlrf.predict(bigTrainX)
Ypredclass1 = 1*(Ypred1 > 0.5)
print("R2 = %f, MSE = %f, Classification Accuracy = %f" % (metrics.r2_score(y1_test, Ypred1), metrics.mean_squared_error(y1_test, Ypred1), metrics.accuracy_score(y1_test, Ypredclass1)))
print("W: ", np.append(np.array(mlrf.intercept_), mlrf.coef_))
'''
#%% - Neural Network
from sklearn.neural_network import MLPRegressor

from datetime import datetime
t1 = datetime.now()
resultsDF = pd.DataFrame(data=None, columns=['Hidden Layer', 'Activation', 'Mean Squared Error', 'Train Accuracy', 'Test Accuracy'])
#firstHL = list(range(150,200))
#secondHL = list(product(range(100,200),repeat=2))

neurons = [10,50,100,150,200]
thirdHL = list(product(neurons,repeat=3))
hiddenLayers = thirdHL

activations = ('relu', 'logistic', 'identity', 'tanh')
# Hidden Layers
for act in activations:
    for hl in hiddenLayers:
        #hl = i
        #currActivate = k
        #regpenalty = 0.0003 #according to forums on the internet, this is an optimal adam solver learning rate
        clf = MLPRegressor(hidden_layer_sizes=(hl)
                            , activation=act
                            , solver='adam'
                            , alpha=0.04
                            , max_iter=10000
                            , validation_fraction=0.42).fit(X_train,y_train)
        annPredY = clf.predict(X_test)
        
        # Get  Scores
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        mse = metrics.mean_squared_error(y_test, annPredY)
        
        print("\n ###### MPL Classifier #######")
        print(f"\n Activation Type: {act}")
        #print(f"\nLearning Rate: {learning_rate}")
        print(f"\nHidden Layers: {hl}")
        print("\n\rANN: MSE = %f" % mse)
        print(f"\nTrain Accuracy = {train_accuracy}")
        print(f"\nTest Accuracy = {test_accuracy}")
   
        resultsDF = resultsDF.append({'Hidden Layer': hl, 
                                      'Activation': act, 
                                      'Mean Squared Error': mse,
                                      'Train Accuracy': train_accuracy,
                                      'Test Accuracy':test_accuracy}, ignore_index=True)

t2 = datetime.now()
total_time = t2-t1
print("Started: ", t1)
print("Ended: ", t2)
print("Total Time for ANN: ", t2-t1)


#%% - Save ANN DF
best_test = resultsDF['Test Accuracy'].nlargest(10)
#best_misslabeled = resultsDF['misclass'].nsmallest(10)
worst_test = resultsDF['Test Accuracy'].nsmallest(10)
#worst_mislabeled = resultsDF['misclass'].nlargest(10)

best_test_df = pd.DataFrame([resultsDF.loc[best_test.index[idx]] for idx in range(len(best_test.index))])
#best_misslabeled_df = pd.DataFrame([resultsDF.loc[best_misslabeled.index[idx]] for idx in range(len(best_misslabeled.index))])
worst_test_df = pd.DataFrame([resultsDF.loc[worst_test.index[idx]] for idx in range(len(worst_test.index))])
#worst_mislabeled_df = pd.DataFrame([resultsDF.loc[worst_mislabeled.index[idx]] for idx in range(len(worst_mislabeled.index))])

best_test_df.to_excel('Best_Test_Accuracy.xlsx')
#best_misslabeled_df.to_excel('Best_Mislabeled_Models.xlsx')
worst_test_df.to_excel('Worst_Test_Accuracy.xlsx')
#worst_mislabeled_df.to_excel('Worst_Mislabeled_Models.xlsx')


#%%
# Ensemble (TY SCIKIT LEARN DOCUMENTATION!)
# Random Forest Bc That Sounds The Most Useful (Did I Profit?)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeRegressor(max_depth=None, min_samples_split=2,random_state=0).fit(X_train, y_train)
predY = clf.predict(X_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())
print(f'Decision Tree \nR2 Score: {metrics.r2_score(y_test, predY)}')
print(f'MSE: {metrics.mean_squared_error(y_test, predY)}')
print(f'Income: {calculate_income(clf, X_norm, df_main)}')

clf = RandomForestRegressor(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_train, y_train)
predY = clf.predict(X_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())
print(f'Random Forest \nR2 Score: {metrics.r2_score(y_test, predY)}')
print(f'MSE: {metrics.mean_squared_error(y_test, predY)}')
print(f'Income: {calculate_income(clf,X_norm, df_main)}')

clf = ExtraTreesRegressor(n_estimators=10, max_depth=None,
                           min_samples_split=2, random_state=0).fit(X_train, y_train)
predY = clf.predict(X_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())
print(f'Extra Tree \nR2 Score: {metrics.r2_score(y_test, predY)}')
print(f'MSE: {metrics.mean_squared_error(y_test, predY)}')
print(f'Income: {calculate_income(clf,X_norm, df_main)}')
#%% Create ANN Best
clf_best = MLPRegressor(hidden_layer_sizes=(200,150,10)
                        , activation='tanh'
                        , solver='adam'
                        , alpha=0.04
                        , max_iter=10000
                        , validation_fraction=0.42).fit(X_train,y_train)
clf_best = clf_best.fit(X_train, y_train.ravel())
annPredY = clf_best.predict(X_norm)

clf_best.predict(X_test)
print(f'R2 Score for ANN: {metrics.r2_score(y_test, clf_best.predict(X_test))}')
print(f'Income: {calculate_income(clf_best,X_norm, df_main)}')
#%%
import matplotlib.pyplot as plt

clf = ExtraTreesRegressor(n_estimators=10, max_depth=None,
                           min_samples_split=2, random_state=0).fit(X_train, y_train)
predY = clf.predict(X_norm)

    
#%%
# Plot 5 Years
plt.figure()
plt.plot(df_main['Date'].values, Y, label = 'Future Stock Price')
plt.plot(df_main['Date'].values, annPredY,label = 'Model Prediction')
plt.plot(df_main['Date'].values, df_main['GM_Close/Last'], label = 'Current Stock Price')
plt.xlabel('Date')
plt.ylabel('GM Stock Price at Close (USD)')
plt.title('Actual and Predictions of Close Stock Price in 28 Days for Past 5 Years \nModel: Extra Trees Regressor')
plt.legend()
plt.show()

## - 3 Months
df_3months = df_main.drop(index = range(63, len(df_main)))
#predictors = df_3months.drop(columns = ['GM_Close/Last_in_28_Days']).columns
#X_3months = df_3months[predictors].to_numpy(np.float64)
#min_max_scaler = MinMaxScaler()
#X_norm_3months = min_max_scaler.fit_transform(X_3months)
#Y_3months = df_3months['GM_Close/Last_in_28_Days'].to_numpy(np.float64)
#print(Y_3months)

# Prediction
Y_3months = Y[:63]
annPredY_3 = clf_best.predict(X_norm[:63])
print(annPredY_3)

# Plot 3 Months
plt.figure()
plt.plot(df_3months['Date'].values, Y_3months, label = 'Future Stock Price')
plt.plot(df_3months['Date'].values, annPredY_3, label = 'Model Prediction')
plt.plot(df_3months['Date'].values, df_main['GM_Close/Last'][:63], label = 'Current Stock Price')
plt.xlabel('Date')
plt.ylabel('GM Stock Price at Close (USD)')
plt.title('Actual and Predictions of Close Stock Price in 28 Days for Past 3 Months \nModel: Extra Trees Regressor')
plt.legend()
plt.show()


