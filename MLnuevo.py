# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:07:35 2021

@author: user
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import statistics as stats


from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

df = pd.read_csv("dataset con estado estrato 2.csv", sep=";",encoding = "ISO-8859-1")
df = pd.read_csv("dataset con estado estrato 3.csv", sep=";",encoding = "ISO-8859-1")
df = pd.read_csv("dataset con estado estrato 4.csv", sep=";",encoding = "ISO-8859-1")
df = pd.read_csv("dataset con estado estrato 5.csv", sep=";",encoding = "ISO-8859-1")
df = pd.read_csv("dataset con estado estrato 6.csv", sep=";",encoding = "ISO-8859-1")

df.head()
df=df.drop('Estrato', axis=1)

cov_mat=df.cov()


#Busqueda de datos vacios
df.isnull().values.any()
len(df)
# Eliminacion de datos vacios
df=df.dropna()
len(df)

#Asignacion de variables
X=df.drop('Precio', axis=1)
y=df['Precio']

#Mapas de Calor
sns.heatmap(df.corr(), annot = True)


#Normalizacion



#Particion del Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


################                  REGRESION LINEAL           ##################
linear_regression= LinearRegression()

linear_regression.fit(X_train,y_train)
y_pred=linear_regression.predict(X_test)

print('Coefficients: \n', linear_regression.coef_)
print('Media: %.2f'% stats.mean(y))
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
RMSE= sqrt(np.mean((y_test - y_pred)**2))
print('RMSE: %.2f'% RMSE)
print('RMSE / Media ')
print(RMSE/stats.mean(y))


plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel('Valor real[1M$]')
plt.ylabel('Prediccion[1M$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.title("Regression Lineal")
plt.show()

sns.regplot(x = y_test, y = y_pred, data = df)

################                  ARBOLES DE DECISION        ##################


tree_reg= DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train,y_train)

y_pred=tree_reg.predict(X_test)

print('Coefficients: \n', linear_regression.coef_)
print('Media: %.2f'% stats.mean(y))
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
RMSE= sqrt(np.mean((y_test - y_pred)**2))
print('RMSE: %.2f'% RMSE)
print('RMSE / Media ')
print(RMSE/stats.mean(y))


plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel('Valor real[1M$]')
plt.ylabel('Prediccion[1M$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.title("Arboles de Decision")
plt.show()
sns.regplot(x = y_test, y = y_pred, data = df)

################                 RANDOM FOREST        ##################

forest_reg=RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(X_train,y_train)
y_pred=forest_reg.predict(X_test)

print('Media: %.2f'% stats.mean(y))
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
RMSE= sqrt(np.mean((y_test - y_pred)**2))
print('RMSE: %.2f'% RMSE)
print('RMSE / Media ')
print(RMSE/stats.mean(y))


plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel('Valor real[1M$]')
plt.ylabel('Prediccion[1M$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.title("Random Forest")
plt.show()
sns.regplot(x = y_test, y = y_pred, data = df)





###############################################################################
###############################################################################
###################  MODELO RED NEURONAL   ####################################
###############################################################################
###############################################################################


##Definiendo modelo


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(8, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer='Adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 80, batch_size = 50)



#history = model.fit_generator(train_generator,steps_per_epoch= steps_per_epoch,
#             epochs=100,validation_data=validation_generator,validation_steps= validation_steps,
#             verbose=2)
#Evaluando Modelo
epochs_hist.history.keys()


#Grafico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del Modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])


test_predictions=model.predict(X_test).flatten()

plt.figure()
plt.scatter(y_test,test_predictions)
plt.xlabel('Valor real[1000$]')
plt.ylabel('Prediccion[1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.show()

sns.regplot(x = y_test, y = test_predictions, data = df)

print('Media: %.2f'% stats.mean(y))
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, test_predictions))
print('Coefficient of determination: %.10f'
      % r2_score(y_test, test_predictions))
RMSE= sqrt(np.mean((y_test - test_predictions)**2))
print('RMSE: %.2f'% RMSE)
print('RMSE / Media ')
print(RMSE/stats.mean(y))

r2=r2_score(y_test, test_predictions)
print('Coeficiente de determinacion = %.2f'% r2)
r2
