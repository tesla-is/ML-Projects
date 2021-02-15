import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Life_Expectancy.csv')

sns.scatterplot(dataset['Adult_Mortality'], dataset['Expected'])

from scipy.stats import pearsonr

pearsonr(dataset['Adult_Mortality'], dataset['Expected'])

X = dataset['Income_Index']
y = dataset['Expected']

import statsmodels.api as sm

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

y_pred = model.predict()

error = y - y_pred
error

residual = np.sum(error)

expected = residual / X.shape[0]
expected

X = dataset.iloc[:, [4, 8, 9]]

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

encoded = pd.get_dummies(dataset['Status'], drop_first = True)

X = dataset.iloc[:, [4, 8, 9]]
X = pd.concat([X, encoded], axis = 1)

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

X = dataset.iloc[:, 2:-1]
X = pd.concat([X, encoded], axis = 1)

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

p_values = pd.DataFrame(model.pvalues, columns = ['p_values'])

p_values[p_values['p_values'] < 0.05]

X = dataset[['GDP', 'Income_Index']]

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

sst = np.sum((y - y.mean()) ** 2)
sst

X = dataset['Income_Index']
model = sm.OLS(y, X).fit()
print(model.summary())

model.conf_int()

X = dataset.iloc[:, [2, 3, 7]]

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

X = dataset.iloc[:, [2, 3, 7, 12]]

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

X = dataset.iloc[:, 2:-1]
X = pd.concat([X, encoded], axis = 1)

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

np.format_float_positional(model.f_pvalue)

from statsmodels.graphics.gofplots import qqplot

qqplot(error, line = 'r')

X['interaction'] = X['Developing'] * X['GDP']

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

X = X.iloc[:, :-1]

model = sm.OLS(y, X).fit()
print(model.summary())

#################################################

dataset = pd.read_csv('LungCapdata.csv')

pd.plotting.scatter_matrix(dataset)
sns.heatmap(dataset.corr(), annot = True)

X = dataset.iloc[:, :-1]

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

y = dataset.iloc[:, -1]

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

y_hat = model.predict()

residual = y - y_hat
residual

residuals = model.resid
residuals

fitted = model.fittedvalues
fitted

sns.scatterplot(fitted, residuals)

from scipy.stats import shapiro
shapiro(residuals)

from sklearn.metrics import mean_squared_error, mean_absolute_error

np.sqrt(mean_squared_error(y, y_hat))
mean_absolute_error(y, y_hat)

def mape(y, y_hat):
    return ((np.sum(np.abs(y - y_hat) / y)) / y.shape[0]) * 100

mape(y, y_hat)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)








































































































































































