
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
data = pd.read_csv('C:\\Users\HP\Desktop\insurance.csv')
x=data.iloc[:,2:3]
y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
"""from sklearn.preprocessing import StandardScaler
object= StandardScaler()
x_train=object.fit_transform(x_train)
x_test=object.transform(x_test)"""
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Expenses vs BMI(tr)')
plt.xlabel('BMI')
plt.ylabel('Expenses')
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(ts)')
plt.xlabel('BMI')
plt.ylabel('Expenses')
plt.show()
print(regressor.predict([[28]]))