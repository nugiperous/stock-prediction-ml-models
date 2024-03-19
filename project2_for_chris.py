
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
#%% Create Binary Target
for idx in df_main.index:
    if df_main.loc[idx, 'GM_Close/Last'] < df_main.loc[idx, 'GM_Close/Last_in_28_Days']:
        df_main.loc[idx, 'Profit'] = True
    else:
        df_main.loc[idx, 'Profit'] = False
    

#%% Imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

#%% Target Data
# Setting up Training Data
# Data
predictors = df_main.drop(columns = ['Profit']).columns
X = df_main[predictors].to_numpy(np.float64)
min_max_scaler = MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)

Y = df_main['Profit'].to_numpy(np.float64)

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

#%%
# Logistic Regression
model = LogisticRegression()
clf_Log = model.fit(X_train, y_train)
Ypred = clf_Log.predict(X_train)
Ypredclass = 1*(Ypred > 0.5)
print("R2 = %f,  MSE = %f,  Classification Accuracy = %f" % (metrics.r2_score(y_test, Ypred), metrics.mean_squared_error(y_test, Ypred), metrics.accuracy_score(y_test, Ypredclass)))
#print("W: ", np.append(np.array(mlr.intercept_), mlr.coef_))
#%%
# Poly
poly = preproc.PolynomialFeatures(2) # object to generate polynomial basis functions
X_train = dataFrame.drop([IDName, targetName], axis=1).to_numpy()
bigTrainX = poly.fit_transform(X_train)
mlrf = linmod.LogisticRegression() # creates the regressor object
mlrf.fit(bigTrainX, y_train)
Ypred = mlrf.predict(bigTrainX)
Ypredclass = 1*(Ypred > 0.5)
print("R2 = %f, MSE = %f, Classification Accuracy = %f" % (metrics.r2_score(y_test, Ypred), metrics.mean_squared_error(y_test, Ypred), metrics.accuracy_score(y_test, Ypredclass)))
print("W: ", np.append(np.array(mlr.intercept_), mlr.coef_))

#%% - Neural Network

import datetime
t1 = datetime.datetime.now()

# Split Testing and Training Data

trainX, testX, trainY, testY = train_test_split(X_norm, Y, test_size=0.3, 
                                                  train_size=0.7, random_state=22222, 
                                                  shuffle=True, stratify=None) 

learning_rate_list = []
hl_1_list = []
hl_2_list = []
hl_3_list = []
roc_auc_score_list = []
mislabeled_list = []
confusion_matrix_list = []
activation_type_list = []

activation_list = ['relu', 'logistic', 'identity', 'tanh']

for activation in activation_list:
    for learning_rate in np.arange(0.0001, 0.1, 0.01): 
        # For MLP with 1 hidden layer
        for hl_1 in range(1, 11):
            # Create the Classifier, fit and predict
            clf = MLPClassifier(hidden_layer_sizes=(hl_1,), activation=activation,
                                solver='adam', alpha=learning_rate, early_stopping=True,
                                validation_fraction=0.42)
            clf.fit(trainX,trainY)
            annPredY = clf.predict(testX)
            
            # Add Stats to List
            activation_type_list.append(activation)
            learning_rate_list.append(learning_rate)
            hl_1_list.append(hl_1)
            hl_2_list.append(0)
            hl_3_list.append(0)
            roc_auc_score_list.append(metrics.roc_auc_score(testY, annPredY))
            mislabeled_list.append((testY != annPredY).sum()/testX.shape[0])
            confusion_matrix_list.append(metrics.confusion_matrix(testY, annPredY))

            print("\n ###### MPL Classifier #######")
            print(f"\n Activation Type: {activation}")
            print(f"\nLearning Rate: {learning_rate}")
            print(f"\nHidden Layers: ({hl_1},)")
            print(f"\n\rANN: AUROC = {metrics.roc_auc_score(testY, annPredY):.6f}")
            print("\n\rANN: %d mislabeled out of %d points" % ((testY != annPredY).sum(), testX.shape[0]))
            print(metrics.confusion_matrix(testY, annPredY))
            
        for hl_1 in range(1, 11):
            for hl_2 in range(1,11):
                # Create the Classifier, fit and predict
                clf = MLPClassifier(hidden_layer_sizes=(hl_1,hl_2), activation=activation,
                                    solver='adam', alpha=learning_rate, early_stopping=True,
                                    validation_fraction=0.42)
                clf.fit(trainX,trainY)
                annPredY = clf.predict(testX)
                 
                # Add Stats to List
                activation_type_list.append(activation)
                learning_rate_list.append(learning_rate)
                hl_1_list.append(hl_1)
                hl_2_list.append(hl_2)
                hl_3_list.append(0)
                roc_auc_score_list.append(metrics.roc_auc_score(testY, annPredY))
                mislabeled_list.append((testY != annPredY).sum()/testX.shape[0])
                confusion_matrix_list.append(metrics.confusion_matrix(testY, annPredY))
    
                print("\n ###### MPL Classifier #######")
                print(f"\n Activation Type: {activation}")
                print(f"\nLearning Rate: {learning_rate}")
                print(f"\nHidden Layers: ({hl_1},{hl_2})")
                print(f"\n\rANN: AUROC = {metrics.roc_auc_score(testY, annPredY):.6f}")
                print("\n\rANN: %d mislabeled out of %d points" % ((testY != annPredY).sum(), testX.shape[0]))
                print(metrics.confusion_matrix(testY, annPredY))
            
        for hl_1 in range(1, 11):
            for hl_2 in range(1,11):    
                for hl_3 in range(1,11):    
                    # Create the Classifier, fit and predict
                    clf = MLPClassifier(hidden_layer_sizes=(hl_1, hl_2, hl_3), activation=activation,
                                        solver='adam', alpha=learning_rate, early_stopping=True,
                                        validation_fraction=0.42)
                    clf.fit(trainX,trainY)
                    annPredY = clf.predict(testX)
                    
                    # Add Stats to List
                    activation_type_list.append(activation)
                    learning_rate_list.append(learning_rate)
                    hl_1_list.append(hl_1)
                    hl_2_list.append(hl_2)
                    hl_3_list.append(hl_3)
                    roc_auc_score_list.append(metrics.roc_auc_score(testY, annPredY))
                    mislabeled_list.append((testY != annPredY).sum()/testX.shape[0])
                    confusion_matrix_list.append(metrics.confusion_matrix(testY, annPredY))
        
                    print("\n ###### MPL Classifier #######")
                    print(f"\n Activation Type: {activation}")
                    print(f"\nLearning Rate: {learning_rate}")
                    print(f"\nHidden Layers: ({hl_1},{hl_2},{hl_3})")
                    print(f"\n\rANN: AUROC = {metrics.roc_auc_score(testY, annPredY):.6f}")
                    print("\n\rANN: %d mislabeled out of %d points" % ((testY != annPredY).sum(), testX.shape[0]))
                    print(metrics.confusion_matrix(testY, annPredY))
            
            
mpl_classifier_stats = pd.DataFrame()
mpl_classifier_stats['Activation Type'] = activation_type_list
mpl_classifier_stats['Learning Rate'] = learning_rate_list
mpl_classifier_stats['HL 1 Neuron Count'] = hl_1_list
mpl_classifier_stats['HL 2 Neuron Count'] = hl_2_list
mpl_classifier_stats['HL 3 Neuron Count'] = hl_3_list
mpl_classifier_stats['AUROC Score'] = roc_auc_score_list
mpl_classifier_stats['Misclassification Rate'] = mislabeled_list
#mpl_classifier_stats['Confusion Matrix'] = confusion_matrix_list
t2 = datetime.datetimenow()
total_time = t2-t1
print("Started: ", t1)
print("Ended: ", t2)
print("Total Time for ANN: ", t2-t1)
#%% Part 5 - Build two tables with ten best and ten worst models

best_auroc = mpl_classifier_stats['AUROC Score'].nlargest(10)
best_misslabeled = mpl_classifier_stats['Misclassification Rate'].nsmallest(10)
worst_auroc = mpl_classifier_stats['AUROC Score'].nsmallest(10)
worst_mislabeled = mpl_classifier_stats['Misclassification Rate'].nlargest(10)


best_auroc_df = pd.DataFrame([mpl_classifier_stats.loc[best_auroc.index[idx]] for idx in range(len(best_auroc.index))])
best_misslabeled_df = pd.DataFrame([mpl_classifier_stats.loc[best_misslabeled.index[idx]] for idx in range(len(best_misslabeled.index))])
worst_auroc_df = pd.DataFrame([mpl_classifier_stats.loc[worst_auroc.index[idx]] for idx in range(len(worst_auroc.index))])
worst_mislabeled_df = pd.DataFrame([mpl_classifier_stats.loc[worst_mislabeled.index[idx]] for idx in range(len(worst_mislabeled.index))])

best_auroc_df.to_excel('Best_Auroc_Models.xlsx')
best_misslabeled_df.to_excel('Best_Mislabeled_Models.xlsx')
worst_auroc_df.to_excel('Worst_Auroc_Models.xlsx')
worst_mislabeled_df.to_excel('Worst_Mislabeled_Models.xlsx')

#%% use best predictor
clf_best = MLPClassifier(hidden_layer_sizes=(9, 8, 8), activation='tanh',
                                        solver='adam', alpha=.0401, early_stopping=True,
                                        validation_fraction=0.42)
clf_best.fit(X_train, y_train.ravel())
annPredY = clf_best.predict(X)

#%% graph
plt.plot(range(1, 1259,annPredY, label = 'Neural Network Predictor')
plt.plot(range(1, 1259,Y, label = 'Neural Network Predictor')

plt.xlabel('Year')
plt.ylabel('MSE')
plt.title('MSE of Each Year')
plt.legend()
plt.show()
#%%
# Ensemble (TY SCIKIT LEARN DOCUMENTATION!)
# Random Forest Bc That Sounds The Most Useful (Should I Invest?)
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() > 0.999

#%% graphing results for model
