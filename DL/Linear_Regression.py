import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('housing_data.csv')
df

df.isnull().sum()

df.fillna(df.mean(), inplace = True)

df.isnull().sum()

df.describe()
df.info()
df.shape

correlation = df.corr()
correlation.loc['MEDV']

sns.histplot(df['MEDV'])

fig,axes = plt.subplots(figsize=(15,12))
sns.heatmap(correlation,square=True,annot=True)

X = df.drop('MEDV',axis=1)
y = df['MEDV']
print(X)
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42, test_size = 0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
r2 = r2_score(y_test,y_pred)
print(r2)

import keras
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = (13)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mae"])
keras.utils.plot_model(model,to_file='model.png',show_shapes = True,show_layer_names = True)
history = model.fit(X_train,y_train,epochs = 100, validation_split = 0.05)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
mse_nn,mae_nn = model.evaluate(X_test,y_test)
print("MSE:",mse_nn)
print("MAE",mae_nn)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:",rmse)

