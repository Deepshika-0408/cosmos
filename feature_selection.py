import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
df = pd.read_excel('C:/Users/DEEPSHIKA/Downloads/cosmos/dataset_for_debris/fengyuan.xlsx')

# Load selected features
with open('final_selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

# Convert 'EPOCH' to datetime and extract features (if not done previously)
df['EPOCH'] = pd.to_datetime(df['EPOCH'], format='%Y-%m-%d %H:%M:%S.%f')

# Prepare data
X = df[selected_features]
y = df['MEAN_MOTION']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM and GRU [samples, time steps, features]
X_train_scaled_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled_reshaped.shape[1], X_train_scaled_reshaped.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))  # Output layer
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
lstm_model.fit(X_train_scaled_reshaped, y_train, validation_data=(X_test_scaled_reshaped, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

# Predictions from LSTM
y_pred_lstm = lstm_model.predict(X_test_scaled_reshaped).flatten()

# Build and train GRU model
gru_model = Sequential()
gru_model.add(GRU(50, activation='relu', input_shape=(X_train_scaled_reshaped.shape[1], X_train_scaled_reshaped.shape[2])))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(1))  # Output layer
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X_train_scaled_reshaped, y_train, validation_data=(X_test_scaled_reshaped, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

# Predictions from GRU
y_pred_gru = gru_model.predict(X_test_scaled_reshaped).flatten()

# Combine predictions as new features for SVM
combined_predictions = np.column_stack((y_pred_lstm, y_pred_gru))

# Train SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(combined_predictions, y_test)

# Make predictions with SVM
y_pred_svm = svm_model.predict(combined_predictions)

# Evaluate the model using Mean Squared Error, R², and Mean Absolute Error
mse = mean_squared_error(y_test, y_pred_svm)
r2 = r2_score(y_test, y_pred_svm)
mae = mean_absolute_error(y_test, y_pred_svm)
print(f'Mean Squared Error of Hybrid Model (LSTM + GRU + SVM): {mse}')
print(f'R² of Hybrid Model (LSTM + GRU + SVM): {r2}')
print(f'Mean Absolute Error of Hybrid Model (LSTM + GRU + SVM): {mae}')

# Optionally, save models
lstm_model.save('lstm_model.h5')
gru_model.save('gru_model.h5')