# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
### Date: 

### AIM:
To implement the SARIMA model using Python for time series analysis on EV (Ethereum) data.


### ALGORITHM:
• Explore the Dataset
• Load the EV dataset, focusing on the time and value columns. Plot the time series to visualize trends.
• Check for Stationarity of Time Series
• Plot the time series data and apply the Augmented Dickey-Fuller (ADF) test to check for stationarity.
• Determine SARIMA Model Parameters (p, d, q, P, D, Q, m)
Use ACF and PACF plots to help estimate the SARIMA parameters.
• Fit the SARIMA Model
Use the SARIMAX model from the statsmodels library with the chosen parameters.
• Make Time Series Predictions
Forecast future values and compare with test data.
• Evaluate Model Predictions
Use Root Mean Squared Error (RMSE) to evaluate prediction accuracy.

### PROGRAM:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')  # Replace with actual path

# Convert the 'time' column to datetime and set it as the index (replace 'time' with your actual column name)
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
data.set_index('time', inplace=True)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['value'], label='EV Data')  # Replace 'value' with actual column name
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('EV Data Time Series')
plt.legend()
plt.show()

# Function to perform ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')

# Check stationarity
check_stationarity(data['value'])  # Replace 'value' with actual column name

# Plot ACF and PACF to determine parameters
plot_acf(data['value'])
plt.title('ACF Plot')
plt.show()

plot_pacf(data['value'])
plt.title('PACF Plot')
plt.show()

# SARIMA model parameters (example values; adjust based on ACF/PACF insights)
p, d, q = 1, 1, 1   # Non-seasonal parameters
P, D, Q, m = 1, 1, 1, 12  # Seasonal parameters (adjust m based on your data frequency)

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['value'][:train_size], data['value'][train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE for evaluation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/e6ec2e8b-037f-4e92-91be-51bba26f11c8)
![image](https://github.com/user-attachments/assets/dfd47978-2cb9-4a96-9c77-6bac70168ae2)
![image](https://github.com/user-attachments/assets/a4ee1b87-11f3-4fea-9d1b-dbbb94259e8b)
![image](https://github.com/user-attachments/assets/e051fb30-fa68-4ba2-9ad8-aa6b58bf81d6)


### RESULT:
The SARIMA model was successfully implemented on the EV dataset for time series forecasting, and the model's performance was evaluated using RMSE.
