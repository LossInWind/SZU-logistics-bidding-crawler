import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_excel('output.xlsx')
data1 = pd.read_excel('output.xlsx', sheet_name=1)

column_data = data.iloc[:, 3:5]
column_data = column_data.dropna()
# print(column_data)
CPI_data = data1.iloc[:, [0, 7]]

## 招标价格统计

data2008 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2008')]
money2008 = data2008['money'].sum() / len(data2008['money'])


data2008_1 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/1')]
data2008_2 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/2')]
data2008_3 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/3')]
#2008年第一个季度
data2008_q1 = pd.concat([data2008_1, data2008_2, data2008_3])
money2008_q1 = data2008_q1['money'].sum()/len(data2008_q1['money'])

data2008_4 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/4')]
data2008_5 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/5')]
data2008_6 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/6')]
#2008年第二个季度
data2008_q2 = pd.concat([data2008_4, data2008_5, data2008_6])
money2008_q2 = data2008_q2['money'].sum()/len(data2008_q2['money'])

data2008_7 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/7')]
data2008_8 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/8')]
data2008_9 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/9')]
#2008年第三个季度
data2008_q3 = pd.concat([data2008_7, data2008_8, data2008_9])
money2008_q3 = data2008_q3['money'].sum()/len(data2008_q3['money'])

data2008_10 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/10')]
data2008_11 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/11')]
data2008_12 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/12')]
#2008年第四个季度
data2008_q4 = pd.concat([data2008_10, data2008_11, data2008_12])
money2008_q4 = data2008_q4['money'].sum()/len(data2008_q4['money'])

data2009 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2009')]
money2009 = data2009['money'].sum() / len(data2009['money'])

data2009_1 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/1')]
data2009_2 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/2')]
data2009_3 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/3')]
#2009年第一个季度
data2009_q1 = pd.concat([data2009_1, data2009_2, data2009_3])
money2009_q1 = data2009_q1['money'].sum()/len(data2009_q1['money'])

data2009_4 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/4')]
data2009_5 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/5')]
data2009_6 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/6')]
#2009年第二个季度
data2009_q2 = pd.concat([data2009_4, data2009_5, data2009_6])
money2009_q2 = data2009_q2['money'].sum()/len(data2009_q2['money'])

data2009_7 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/7')]
data2009_8 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/8')]
data2009_9 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/9')]
#2009年第三个季度
data2009_q3 = pd.concat([data2009_7, data2009_8, data2009_9])
money2009_q3 = data2009_q3['money'].sum()/len(data2009_q3['money'])

data2009_10 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/10')]
data2009_11 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/11')]
data2009_12 = data2009[data2009.iloc[:, 0].astype(str).str.startswith('2009/12')]
#2009年第四个季度
data2009_q4 = pd.concat([data2009_10, data2009_11, data2009_12])
money2009_q4 = data2009_q4['money'].sum()/len(data2009_q4['money'])

datalen_2008 = len(data2008['money'])
datalen_2008_q1 = len(data2008_q1['money'])
datalen_2008_q2 = len(data2008_q2['money'])
datalen_2008_q3 = len(data2008_q3['money'])
datalen_2008_q4 = len(data2008_q4['money'])
datalen_2009 = len(data2009['money'])
datalen_2009_q1 = len(data2009_q1['money'])
datalen_2009_q2 = len(data2009_q2['money'])
datalen_2009_q3 = len(data2009_q3['money'])
datalen_2009_q4 = len(data2009_q4['money'])


data = pd.read_excel('output.xlsx')
data1 = pd.read_excel('output.xlsx', sheet_name=1)
column_data = data.iloc[:, 3:5]
column_data = column_data.dropna()
CPI_data = data1.iloc[:, [0, 7]]

datalen = {}

for year in range(2005, 2024):
    data_year = column_data[column_data.iloc[:, 0].astype(str).str.startswith(str(year))]
    datalen[f'{year}'] = len(data_year['money'])
    for quarter in range(1, 5):
        months = [f'{year}/{month}' for month in range((quarter-1)*3+1, quarter*3+1)]
        data_quarter = pd.concat([data_year[data_year.iloc[:, 0].astype(str).str.startswith(month)] for month in months])
        datalen[f'{year}_q{quarter}'] = len(data_quarter['money'])

# Convert the dictionary to a DataFrame and write it to an Excel file
datalen_df = pd.DataFrame(list(datalen.items()), columns=['Year/Quarter', 'Length'])
datalen_df.to_excel('datalen.xlsx', index=False)


