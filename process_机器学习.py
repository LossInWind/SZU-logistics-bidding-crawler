import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
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
moneysum_2008 = data2008['money'].sum()

data2009 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2009')]
money2009 = data2009['money'].sum() / len(data2009['money'])
moneysum_2009 = data2009['money'].sum()

data2010 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2010')]
money2010 = data2010['money'].sum() / len(data2010['money'])
moneysum_2010 = data2010['money'].sum()

data2011 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2011')]
money2011 = data2011['money'].sum() / len(data2011['money'])
moneysum_2011 = data2011['money'].sum()

data2012 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2012')]
money2012 = data2012['money'].sum() / len(data2012['money'])
moneysum_2012 = data2012['money'].sum()

data2013 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2013')]
money2013 = data2013['money'].sum() / len(data2013['money'])
moneysum_2013 = data2013['money'].sum()

data2014 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2014')]
money2014 = data2014['money'].sum() / len(data2014['money'])
moneysum_2014 = data2014['money'].sum()

data2015 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2015')]
money2015 = data2015['money'].sum() / len(data2015['money'])
moneysum_2015 = data2015['money'].sum()

data2016 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2016')]
money2016 = data2016['money'].sum() / len(data2016['money'])
moneysum_2016 = data2016['money'].sum()

data2017 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2017')]
money2017 = data2017['money'].sum() / len(data2017['money'])
moneysum_2017 = data2017['money'].sum()

data2018 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2018')]
money2018 = data2018['money'].sum() / len(data2018['money'])
moneysum_2018 = data2018['money'].sum()

data2019 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2019')]
money2019 = data2019['money'].sum() / len(data2019['money'])
moneysum_2019 = data2019['money'].sum()

data2020 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2020')]
money2020 = data2020['money'].sum() / len(data2020['money'])
moneysum_2020 = data2020['money'].sum()

data2021 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2021')]
money2021 = data2021['money'].sum() / len(data2021['money'])
moneysum_2021 = data2021['money'].sum()

data2022 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2022')]
money2022 = data2022['money'].sum() / len(data2022['money'])
moneysum_2022 = data2022['money'].sum()

data2023 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2023')]
money2023 = data2023['money'].sum() / len(data2023['money'])
moneysum_2023 = data2023['money'].sum()

## CPI汇总 假设一开始为100
CPI2008 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2008')]
for x in CPI2008.iloc[:, 1]:
    result2008 = 100 * (1 + x)

CPI2009 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2009')]
for x in CPI2009.iloc[:, 1]:
    result2009 = result2008 * (1 + x)

CPI2010 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2010')]
for x in CPI2010.iloc[:, 1]:
    result2010 = result2009 * (1 + x)

CPI2011 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2011')]
for x in CPI2011.iloc[:, 1]:
    result2011 = result2010 * (1 + x)

CPI2012 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2012')]
for x in CPI2012.iloc[:, 1]:
    result2012 = result2011 * (1 + x)

CPI2013 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2013')]
for x in CPI2013.iloc[:, 1]:
    result2013 = result2012 * (1 + x)

CPI2014 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2014')]
for x in CPI2014.iloc[:, 1]:
    result2014 = result2013 * (1 + x)

CPI2015 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2015')]
for x in CPI2015.iloc[:, 1]:
    result2015 = result2014 * (1 + x)

CPI2016 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2016')]
for x in CPI2016.iloc[:, 1]:
    result2016 = result2015 * (1 + x)

CPI2017 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2017')]
for x in CPI2017.iloc[:, 1]:
    result2017 = result2016 * (1 + x)

CPI2018 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2018')]
for x in CPI2018.iloc[:, 1]:
    result2018 = result2017 * (1 + x)

CPI2019 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2019')]
for x in CPI2019.iloc[:, 1]:
    result2019 = result2018 * (1 + x)

CPI2020 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2020')]
for x in CPI2020.iloc[:, 1]:
    result2020 = result2019 * (1 + x)

CPI2021 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2021')]
for x in CPI2021.iloc[:, 1]:
    result2021 = result2020 * (1 + x)

CPI2022 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2022')]
for x in CPI2022.iloc[:, 1]:
    result2022 = result2021 * (1 + x)

CPI2023 = CPI_data[CPI_data.iloc[:, 0].astype(str).str.startswith('2023')]
for x in CPI2023.iloc[:, 1]:
    result2023 = result2022 * (1 + x)

Y = np.array(
    [money2008, money2009, money2010, money2011, money2012, money2013, money2014, money2015, money2016, money2017,
     money2018, money2019, money2020, money2021, money2022])
X = np.array(
    [result2008, result2009, result2010, result2011, result2012, result2013, result2014, result2015, result2016,
     result2017, result2018, result2019, result2020, result2021, result2022])

## 模型
model1 = DecisionTreeRegressor()
model2 = LinearRegression()
model3 = RandomForestRegressor()

## 拟合
model1.fit(X.reshape(-1, 1), Y)
model2.fit(X.reshape(-1, 1), Y)
model3.fit(X.reshape(-1, 1), Y)

## 预测
X_new = np.array(result2023).reshape(-1, 1)
Y_pre1 = model1.predict(X_new)
Y_pre2 = model2.predict(X_new)
Y_pre3 = model3.predict(X_new)
Y_true = money2023

print('决策树预测结果：', Y_pre1, '真实结果：', Y_true, "误差百分比", (Y_pre1 - Y_true) / Y_true)
print('线性回归预测结果：', Y_pre2, '真实结果：', Y_true, "误差百分比", (Y_pre2 - Y_true) / Y_true)
print('随机森林预测结果：', Y_pre3, '真实结果：', Y_true, "误差百分比", (Y_pre3 - Y_true) / Y_true)

# Modify the code to use years as the independent variable
years = np.array([2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])

# Modify the code to use years as the independent variable
X = years
Y = np.array(
    [money2008, money2009, money2010, money2011, money2012, money2013, money2014, money2015, money2016, money2017,
     money2018, money2019, money2020, money2021, money2022])

z = np.array(
    [result2008, result2009, result2010, result2011, result2012, result2013, result2014, result2015, result2016,
     result2017, result2018, result2019, result2020, result2021, result2022])

Moneysum = np.array([moneysum_2008, moneysum_2009, moneysum_2010, moneysum_2011, moneysum_2012, moneysum_2013,
                    moneysum_2014, moneysum_2015, moneysum_2016, moneysum_2017, moneysum_2018, moneysum_2019,
                    moneysum_2020, moneysum_2021, moneysum_2022])

# 把XY写入excel文件
df = pd.DataFrame({'year': X, 'money_0': Y, 'cpi': z})
df.to_excel('项目单价变化趋势.xlsx', index=False)

df = pd.DataFrame({'year': X,  'Moneysum': Moneysum, 'cpi': z})
df.to_excel('项目总价变化趋势.xlsx', index=False)

# Fit the regression models
model1.fit(X.reshape(-1, 1), Y)
model2.fit(X.reshape(-1, 1), Y)
model3.fit(X.reshape(-1, 1), Y)

# Modify the code to predict tender prices for 2023
X_new = np.array([2023]).reshape(-1, 1)
Y_pre1 = model1.predict(X_new)
Y_pre2 = model2.predict(X_new)
Y_pre3 = model3.predict(X_new)
Y_true = money2023

print('Decision Tree prediction for 2023:', Y_pre1, 'True value:', Y_true, 'Errorprecent:', (Y_pre1 - Y_true) / Y_true)
print('Linear Regression prediction for 2023:', Y_pre2, 'True value:', Y_true, 'Errorprecent:',
      (Y_pre2 - Y_true) / Y_true)
print('Random Forest prediction for 2023:', Y_pre3, 'True value:', Y_true, 'Errorprecent:', (Y_pre3 - Y_true) / Y_true)
