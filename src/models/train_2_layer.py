import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(
    '../../data/processed/household_power_consumption_processed.csv',
    parse_dates=['Datetime'],
    index_col='Datetime'
    )


results_dic = Path('../../analysis/results')
pics_dic = Path('../../analysis/pics')

results_dic.mkdir(parents=True, exist_ok=True)
pics_dic.mkdir(parents=True, exist_ok=True)



target = "Daily_Energy_KWh"

X = df.copy()
y = df[target]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

look_back = 365

def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

X_train_lstm, y_train_lstm = create_sequences(
    X_train_scaled,
    y_train_scaled,
    look_back
)


X_test_input = np.concatenate(
    [
        X_train_scaled[-look_back:],
        X_test_scaled
    ],
    axis=0
)

y_test_input = np.concatenate(
    [
        y_train_scaled[-look_back:],
        y_test_scaled
    ],
    axis=0
)

X_test_lstm, y_test_lstm = create_sequences(
    X_test_input,
    y_test_input,
    look_back
)

print(X_train_lstm.shape)
print(y_train_lstm.shape)

print(X_test_lstm.shape)
print(y_test_lstm.shape)

model = Sequential()

model.add(
    LSTM(
        units=128,
        return_sequences=True,
        input_shape=(look_back, X_train_lstm.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=64,
        return_sequences=False
    )
)

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()

history = model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

training_time = time.time() - start_time


plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")

plt.legend()
plt.grid(True)

plt.savefig(
    pics_dic / "LSTM_128_64_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

y_pred = model.predict(X_test_lstm)

# Calculate metrics

mae = mean_absolute_error(
    y_test_lstm,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_lstm,
        y_pred
    )
)

r2 = r2_score(
    y_test_lstm,
    y_pred
)


# Save experiment results

results = pd.DataFrame({
    "Model": ["LSTM_128_64"],
    "Layers": [1],
    "Units": ["64"],
    "Train Loss": [history.history["loss"][-1]],
    "Validation Loss": [history.history["val_loss"][-1]],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "Epochs": [len(history.history["loss"])],
    "Training Time (s)": [round(training_time, 2)]
})


results.to_csv(
    results_dic / "LSTM_128_64.csv",
    index=False
)


print(results)