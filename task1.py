#This is a sample Python script.
#Author:Sathvika S
#DATA-SET interpretation and implementing linear regeression model


#importing liraries to get dataset and fit the linear regression model
import pandas as pand
import matplotlib.pyplot as mlib
data=pand.read_csv('http://bit.ly/w-data')       #reading the dataset given using pandas method-'read_csv'

print("")
score_data=data.rename(columns={'Hours':'No_of_hours','Scores':'Scored_marks'},inplace=False)  #changing the column names
print(score_data)    #print the obtained data

#divide the data to test and train the model

from sklearn.model_selection import train_test_split

X=score_data.iloc[:, :-1].values
Y=score_data.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 5)

#to print the trained data-set
print("-----------------------------------------------------The training data is-----------------------------------------------------")
print("Study_hours:",X_train.reshape(1,20))
print("Scores:",Y_train)

#Plot the trained dataset

mlib.scatter(X_train,Y_train, color = 'red',marker='D')
mlib.title('Plot of the trained data set')
mlib.xlabel('Hours_studied')
mlib.ylabel('Marks')
mlib.show()


# Fitting Simple Linear Regression to the given Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

#the Training set results
mlib.scatter(X_train, Y_train, color = 'red',marker='*')
mlib.plot(X_train, model.predict(X_train), color = 'blue')
mlib.title('Linear Regression model')
mlib.xlabel('Hours_studied')
mlib.ylabel('Marks')
mlib.show()

#to predict for the given input
print("")
print("Enter the hours of study to predict the marks as per the linear regression model")
inp =float(input())
ans=model.predict([[inp]])
print("")
print("The predicted score for the given input is :-",ans)

#to find the difference between the actual and predicted values[by the fit]
from sklearn import metrics
print("")
diff=pand.DataFrame({"    acutal values": Y_test,"      predicted values": Y_pred, "     Difference between them": Y_pred-Y_test})
print(diff)
print("")
print("")
print('The Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))
