import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from models.LSTM import *
import os
data_dir = "./datasets/commodity_1_district_3.csv"

# Loading Dataset and splitting it into train and test set
df = pd.read_csv(data_dir, parse_dates=['date'], index_col='date')
# df = df.groupby('date').mean()
df = df.groupby('date')['price'].mean().to_frame()
df = df.sort_index()
df = df.asfreq('D')
df['price'] = df['price'].interpolate(method='time')
print(df.head())

series = df['price'].astype(float)  # just to ensure float type
train_size = int(len(series) * 0.8)
X_train, X_test = series[:train_size], series[train_size:]

# Training SARIMA model to learn the linear patterns
sarima_model = SARIMAX(X_train, order=(1,1,1), seasonal_order=(1,1,1,12))  # order = (p,d,q) seasonal_order = (p,d,q,s)
sarima_fit = sarima_model.fit(disp=False)
sarima_pred = sarima_fit.predict(start=len(X_train), end=len(series)-1)
print("SARIMA model trained.")

# Training LSTM model to learn the non-linear patterns
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(X_train.values.reshape(-1, 1))
scaled_test = scaler.transform(X_test.values.reshape(-1, 1))

window = 12
X_train, y_train = create_sequences(scaled_train, window)
X_test, y_test = create_sequences(scaled_test, window)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
if X_train_tensor.ndim == 2:
    X_train_tensor = X_train_tensor.unsqueeze(-1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
if X_test_tensor.ndim == 2:
    X_test_tensor = X_test_tensor.unsqueeze(-1)


losses = []
for epoch in range(10):
    print(f"epoch {epoch}")
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output.squeeze(), y_train_tensor)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()

model.eval()
with torch.no_grad():
    lstm_pred_scaled = model(X_test_tensor).squeeze().numpy()

lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()


# Pairing the values for both the models
sarima_pred = sarima_pred[-len(lstm_pred):]
true_values = X_test[-len(lstm_pred):]


# Using XGBoost to create an ensemble
ensemble_X = np.vstack([sarima_pred, lstm_pred]).T
ensemble_y = y_test[-len(lstm_pred):].reshape(-1)

xgb_model = XGBRegressor(n_estimators=400)
xgb_model.fit(ensemble_X, ensemble_y)
final_pred = xgb_model.predict(ensemble_X)

# Plotting loss 
rmse = np.sqrt(mean_squared_error(ensemble_y, final_pred))
print(f"Final RMSE: {rmse:.2f}")
plt.figure(figsize=(12,6))
plt.plot(series.index[-len(final_pred):], ensemble_y, label='True')
plt.plot(series.index[-len(final_pred):], final_pred, label='Final Prediction (XGBoost)')
plt.legend()
plt.title("Hybrid SARIMA + LSTM + XGBoost Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the vlaues and predictions and the log loss
true_values = ensemble_y.reshape(-1)
predictions = final_pred.reshape(-1)

np.save("true_values.npy", true_values)
np.save("predictions.npy", final_pred)
np.save("training_loss.npy", np.array(losses))

