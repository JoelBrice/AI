"""""""""""
Regression 

"""""""""""

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

x = Boston_P.data

y = Boston_P.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size= 0.75,
                                                    random_state=76)

from sklearn.preprocessing import MinMaxScaler

Sc = MinMaxScaler(feature_range = (0,1))

X_train = Sc.fit_transform(X_train)

X_test = Sc.fit_transform(X_test)

y_train = y_train.reshape(-1,1)

y_train = Sc.fit_transform(y_train)


"""""""""""
MLR 

"""""""""""

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()

Linear_R.fit(X_train,y_train)

Predicted_values_MLR = Linear_R.predict(X_test)

Predicted_values_MLR = Sc.inverse_transform(Predicted_values_MLR)


"""""""""""
Evaluation Metrics 

"""""""""""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAE = mean_absolute_error(y_test,Predicted_values_MLR)

MSE = mean_squared_error(y_test,Predicted_values_MLR)

RMSE = math.sqrt(MSE)

R2 = r2_score(y_test,Predicted_values_MLR)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

MAPE = mean_absolute_percentage_error(y_test, Predicted_values_MLR)

"""""""""""
PLR 

"""""""""""
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

x = Boston_P.data[:,5]

y = Boston_P.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size= 0.75,
                                                    random_state=76)

from sklearn.preprocessing import PolynomialFeatures

Poly_P = PolynomialFeatures(degree=2)

X_train = X_train.reshape(-1,1)

Poly_X = Poly_P.fit_transform(X_train)

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()

Poly_L_R = Linear_R.fit(Poly_X,y_train)

X_test = X_test.reshape(-1,1)

Poly_Xt = Poly_P.fit_transform(X_test)

Predicted_value_P = Poly_L_R.predict(Poly_Xt)

from sklearn.metrics import r2_score

R2 = r2_score(y_test,Predicted_value_P)


"""""""""""
Random Forest

"""""""""""

from sklearn.ensemble import RandomForestRegressor

Random_F = RandomForestRegressor(n_estimators = 500, max_depth = 20, random_state=33)

Random_F.fit(X_train,y_train)

Predicted_Val_RF = Random_F.predict(X_test)

Predicted_Val_RF = Predicted_Val_RF.reshape(-1,1)

Predicted_Val_RF = Sc.inverse_transform(Predicted_Val_RF)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAE = mean_absolute_error(y_test,Predicted_Val_RF)

MSE = mean_squared_error(y_test,Predicted_Val_RF)

RMSE = math.sqrt(MSE)

R2 = r2_score(y_test,Predicted_Val_RF)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

MAPE = mean_absolute_percentage_error(y_test, Predicted_Val_RF)

"""""""""""
SVR

"""""""""""

from sklearn.svm import SVR

Regressor_SVR = SVR(kernel='rbf')

Regressor_SVR.fit(X_train,y_train)

Predicted_values_SVR = Regressor_SVR.predict(X_test)

Predicted_values_SVR = Predicted_values_SVR.reshape(-1,1)


Predicted_values_SVR = Sc.inverse_transform(Predicted_values_SVR)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAE = mean_absolute_error(y_test,Predicted_values_SVR)

MSE = mean_squared_error(y_test,Predicted_values_SVR)

RMSE = math.sqrt(MSE)

R2 = r2_score(y_test,Predicted_values_SVR)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

MAPE = mean_absolute_percentage_error(y_test, Predicted_values_SVR)










