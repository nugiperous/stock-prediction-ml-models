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

