# EXPERIMENT 02: IMPLEMENTATION OF SIMPLE LINEAR REGRESSION MODEL FOR PREDICTING THE MARKS SCORED
## AIM:  
To write a program to predict the marks scored by a student using the simple linear regression model.
## EQUIPMENTS REQUIRED:
1. Hardware – PCs  
2. Anaconda – Python 3.7 Installation / Jupyter notebook  
## ALGORITHM:
1.Import the required libraries and read the dataframe.  
2.Assign hours to X and scores to Y.  
3.Implement training set and test set of the dataframe.  
4.Plot the required graph both for test data and training data.  
5.Find the values of MSE , MAE and RMSE.     

## PROGRAM:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RITHIGA SRI.B
RegisterNumber:  212221230083
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#Displaying the content in datafile
df.head()

#Last five rows
df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y

#Splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#Displaying predicted values
Y_pred

#Displaying actual values
Y_test

#Graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="orange")
plt.title("Hours VS Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#MSE
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

#MAE
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

#RMSE
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## OUTPUT:
* df.head():  
![image](https://user-images.githubusercontent.com/93427256/229011108-3537a1cb-f91d-49a9-b433-0d28f4857d07.png)
* df.tail():
![image](https://user-images.githubusercontent.com/93427256/229011227-7495679f-df5b-4d34-94ae-7b190508d238.png)
* Array Value of X:    
![image](https://user-images.githubusercontent.com/93427256/229011551-4729446c-b05d-45ea-8a01-f311ea3645ff.png)  
* Array value of Y:
![image](https://user-images.githubusercontent.com/93427256/229011607-c91a9f24-8f7b-4764-b9e3-b546fb8bc295.png)
* Values of Y Prediction:    
![image](https://user-images.githubusercontent.com/93427256/229011701-9f788ade-f430-4f5b-a2f0-747b44843fda.png)
* Array values of Y test:  
![image](https://user-images.githubusercontent.com/93427256/229011934-8a9f4052-33fd-47f3-9cf4-15a4c65f34ba.png)
* Training Set Graph:  
![image](https://user-images.githubusercontent.com/93427256/229012045-6bb0b9a2-32e1-44df-8314-2e8d8528676b.png)
* Test Set Graph:    
![image](https://user-images.githubusercontent.com/93427256/229012148-ba27f12b-36d4-4c7b-8742-c2100b0cc6c7.png)
* VAlues of MSE,MAE and RMSE:  
![image](https://user-images.githubusercontent.com/93427256/229012231-90ffefdb-f0b2-40a2-a8fd-3afd55f99c18.png)

## RESULT:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
