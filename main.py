import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('headbrain.csv')
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
# mean of X and Y
mean_x = data['Head Size(cm^3)'].mean()
mean_y = data['Brain Weight(grams)'].mean()
# total number of rows
n = len(X)
numerator = 0
denuminator = 0
for i in range(n):
    numerator = numerator + ((X[i] - mean_x)*(Y[i] - mean_y))
    denuminator = denuminator + (X[i] - mean_x)**2
m = numerator / denuminator
c = (mean_y - (m * mean_x))
min_x = np.min(X)
max_x = np.max(X)
x = np.linspace(min_x, max_x)
y = (m * x) + c
plt.plot(x, y, color='red', label='linear regression', linewidth=3)
plt.scatter(X, Y, color='blue', label='Scatter Plot', alpha=0.5)
plt.xlabel('Head size (cm^3)')
plt.ylabel('Brain weight(grams)')
plt.legend()
plt.show()
RSS = 0
TSS = 0
for i in range(n):
    y_prediction = (m * X[i]) + c
    RSS += (Y[i] - y_prediction)**2
    TSS += (Y[i] - mean_y)**2
R_square = 1 - (RSS / TSS)
print(f'the goodness of fitting is {R_square} percent')

