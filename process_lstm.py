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

# data2008_1 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/1')]
# data2008_2 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/2')]
# data2008_3 = data2008[data2008.iloc[:, 0].astype(str).str.startswith('2008/3')]
# #2008年第一个季度
# data2008_q1 = pd.concat([data2008_1, data2008_2, data2008_3])
# money2008_q1 = data2008_q1['money'].sum()/len(data2008_q1['money'])

data2009 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2009')]
money2009 = data2009['money'].sum() / len(data2009['money'])

data2010 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2010')]
money2010 = data2010['money'].sum() / len(data2010['money'])

data2011 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2011')]
money2011 = data2011['money'].sum() / len(data2011['money'])

data2012 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2012')]
money2012 = data2012['money'].sum() / len(data2012['money'])

data2013 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2013')]
money2013 = data2013['money'].sum() / len(data2013['money'])

data2014 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2014')]
money2014 = data2014['money'].sum() / len(data2014['money'])

data2015 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2015')]
money2015 = data2015['money'].sum() / len(data2015['money'])

data2016 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2016')]
money2016 = data2016['money'].sum() / len(data2016['money'])

data2017 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2017')]
money2017 = data2017['money'].sum() / len(data2017['money'])

data2018 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2018')]
money2018 = data2018['money'].sum() / len(data2018['money'])

data2019 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2019')]
money2019 = data2019['money'].sum() / len(data2019['money'])

data2020 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2020')]
money2020 = data2020['money'].sum() / len(data2020['money'])

data2021 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2021')]
money2021 = data2021['money'].sum() / len(data2021['money'])

data2022 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2022')]
money2022 = data2022['money'].sum() / len(data2022['money'])

data2023 = column_data[column_data.iloc[:, 0].astype(str).str.startswith('2023')]
money2023 = data2023['money'].sum() / len(data2023['money'])

# Modify the code to use years as the independent variable
years = torch.tensor([2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).view(
    -1, 1)

# Modify the code to use average money values as the target variable
Y = torch.tensor(
    [money2008, money2009, money2010, money2011, money2012, money2013, money2014, money2015, money2016, money2017,
     money2018, money2019, money2020, money2021, money2022]).view(-1, 1)


# Prepare the input data and target variable

# Prepare the input data and target variable
def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequence = data[i:i + window_size]
        target = data[i + window_size]
        if len(sequence) == window_size:
            sequences.append(sequence)
            targets.append(target)
    return torch.stack(sequences).unsqueeze(2), torch.stack(targets)




# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[-1])
        return output


# Convert the data to PyTorch tensors
X_tensor = years.float()
data = Y_tensor = Y.float()
# Convert data into sequences and targets
window_size = 3
X_sequences, Y_targets = create_sequences(data, window_size)

# Define model parameters
input_size = 1
hidden_size = 50
output_size = 1

# Create the LSTM model
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1500)

# Convert the data to PyTorch tensors
X_sequences = X_sequences.float()
Y_targets = Y_targets.float()

# Set the number of training epochs
num_epochs = len(X_sequences)

# Train the LSTM model
for epoch in range(num_epochs):
    model.zero_grad()
    X_sequence = X_sequences[epoch].float()
    Y_target = Y_targets[epoch].float()
    output = model(X_sequence)
    loss = criterion(output, Y_target)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')


# Prepare the input tensor for the next prediction
X_new_sequence = torch.tensor([33455.6562, 31377.7773, 57084.3008]).float().unsqueeze(1).unsqueeze(1)

# Make predictions for the next value
model.eval()
with torch.no_grad():
    Y_pred = model(X_new_sequence)

print('LSTM prediction for the next value:', Y_pred.item(),'True value:',money2023,"误差百分比",(Y_pred.item()-money2023)/money2023)