import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

true_values = np.load("true_values.npy")
predictions = np.load("predictions.npy")
true_values = true_values.reshape(-1)
predictions = predictions.reshape(-1)


mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
r2 = r2_score(true_values, predictions)

print(f"Model Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
