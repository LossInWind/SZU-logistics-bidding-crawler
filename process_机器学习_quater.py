import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Read the data from the Excel file
data = pd.read_excel('output.xlsx')
data1 = pd.read_excel('output.xlsx', sheet_name=1)

# Extract the relevant columns for analysis
column_data = data.iloc[:, 3:5]
column_data = column_data.dropna()

# Extract the CPI data
CPI_data = data1.iloc[:, [0, 7]]

# Define the quarters and years
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Calculate money statistics by quarter
money_values = []

for year in years:
    data_year = column_data[column_data.iloc[:, 0].astype(str).str.startswith(str(year))]

    for i in range(1, 5):
        start_month = (i - 1) * 3 + 1
        end_month = i * 3
        quarters_data = data_year[data_year.iloc[:, 0].astype(str).str.startswith(f'{year}/{start_month}')]

        money_sum = quarters_data['money'].sum()
        money_mean = money_sum / len(quarters_data['money'])

        money_values.append(money_mean)

        print(f"{year} {quarters[i - 1]} - Sum: {money_sum}, Mean: {money_mean}")

# Calculate cumulative CPI by quarter
cpi_values = []

for year in years:
    cumulative_cpi_by_quarter = []

    for i in range(1, 5):
        cumulative_cpi = 100

        for j in range(3):
            month = i * 3 - j
            month_cpi = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith(f'{year}年{month:02d}月份')]

            for x in month_cpi.iloc[:, 1]:
                cumulative_cpi *= (1 + x)

        cumulative_cpi_by_quarter.extend([cumulative_cpi])

    cpi_values.extend(cumulative_cpi_by_quarter)

# Convert the lists to numpy arrays
Y = np.array(money_values[:60])
X = np.array(cpi_values[:60])

# Print the arrays
print("\nMoney values:")
print(Y)
print("\nCumulative CPI values:")
print(X)

## 模型
model1 = DecisionTreeRegressor()
model2 = LinearRegression()
model3 = RandomForestRegressor()

## 拟合
model1.fit(X.reshape(-1, 1), Y)
model2.fit(X.reshape(-1, 1), Y)
model3.fit(X.reshape(-1, 1), Y)

## 预测
result2023 = cpi_values[60:61]
money2023 = money_values[60:61]
X_new = np.array(result2023).reshape(-1, 1)
Y_pre1 = model1.predict(X_new)
Y_pre2 = model2.predict(X_new)
Y_pre3 = model3.predict(X_new)
Y_true = money2023

print('决策树预测结果：', Y_pre1, '真实结果：', Y_true,"误差百分比" , (Y_pre1-Y_true)/Y_true)
print('线性回归预测结果：', Y_pre2, '真实结果：', Y_true,"误差百分比" , (Y_pre2-Y_true)/Y_true)
print('随机森林预测结果：', Y_pre3, '真实结果：', Y_true,"误差百分比" , (Y_pre3-Y_true)/Y_true)




# Modify the code to use years as the independent variable

# Modify the code to use years as the independent variable
x = np.arange(1, 61)
Y = np.array(money_values[:60])

# Fit the regression models
model1.fit(X.reshape(-1, 1), Y)
model2.fit(X.reshape(-1, 1), Y)
model3.fit(X.reshape(-1, 1), Y)

# Modify the code to predict tender prices for 2023
X_new = np.array([62]).reshape(-1, 1)
Y_pre1 = model1.predict(X_new)
Y_pre2 = model2.predict(X_new)
Y_pre3 = model3.predict(X_new)
Y_true = money2023

print('Decision Tree prediction for 2023:', Y_pre1, 'True value:', Y_true, "误差百分比" , (Y_pre1-Y_true)/Y_true)
print('Linear Regression prediction for 2023:', Y_pre2, 'True value:', Y_true, "误差百分比" , (Y_pre2-Y_true)/Y_true)
print('Random Forest prediction for 2023:', Y_pre3, 'True value:', Y_true, "误差百分比" , (Y_pre3-Y_true)/Y_true)

