import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from torch import nn, optim

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

data = Y_tensor = torch.tensor(Y).float()
# Convert data into sequences and targets
window_size = 3
X_sequences, Y_targets = create_sequences(data, window_size)

# Define model parameters
input_size = 1
hidden_size = 5
output_size = 1

# Create the LSTM model
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=6800)

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
test = money_values[57:60]
X_new_sequence = torch.tensor(test).float().unsqueeze(1).unsqueeze(1)
money2023 = money_values[60]
# Make predictions for the next value
model.eval()
with torch.no_grad():
    Y_pred = model(X_new_sequence)

print('LSTM prediction for the next value:', Y_pred.item(), 'True value:', money2023,"误差百分比：",(Y_pred.item()-money2023)/money2023*100,"%")
