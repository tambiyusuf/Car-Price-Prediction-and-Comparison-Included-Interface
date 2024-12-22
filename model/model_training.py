import pandas as pd
import numpy as np
import os 
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras import Sequential
from keras import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.abspath(__file__))
data_path =  os.path.join(project_root,"data", "pre_model.csv")

pre_model = pd.read_csv(data_path)

X = pre_model.drop(columns=['target'])  # Hedef değişkeni dışındaki tüm sütunlar
y = pre_model['target']  # Hedef değişken

# Veriyi normalize etme
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Yapay Sinir Ağı Modeli
model = Sequential()

# İlk gizli katman
model.add(Dense(units=512, activation='relu', input_dim=X_train.shape[1]))

# Dropout katmanı (overfitting'i engellemek için)
model.add(Dropout(0.2))

# İkinci gizli katman
model.add(Dense(units=256, activation='relu'))

# Dropout katmanı
model.add(Dropout(0.2))

# üçüncü gizli katman
model.add(Dense(units=128, activation='relu'))

# Dropout katmanı
model.add(Dropout(0.2))

# Çıkış katmanı (regresyon için tek bir değer)
model.add(Dense(units=1))

# Modeli derleme (regresyon için 'mean_squared_error' kaybı kullanılır)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Modelin test verisi ile tahmin yapma
y_test_pred = model.predict(X_test)

#____________________Testing Model__________________#
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print(f"R² Skoru: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

#____________________Saving Weights and Models__________________#

model.save('last_model.keras')  
joblib.dump(scaler, 'scaler.pkl') 
