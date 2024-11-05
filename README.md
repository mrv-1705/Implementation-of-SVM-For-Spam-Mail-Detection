# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Read the required csv file and store the dataset in a variable.

Step 3: Shape the data and specify x and y value.

Step 4: Split the data for training and testing. 

Step 5: Carry out the preprocessing techniques.

Step 6: Train the dataset using svc.

Step 7: Find y prediction value, accuracy score, confusion matrix and classification report.

Step 8: End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Cynthia Mehul J
RegisterNumber: 212223240020
*/
import pandas as pd 
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding='Windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("y prediction")
print(y_pred)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print("accuracy:")
print(acc)
con=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(con)
c1=classification_report(y_test,y_pred)
print("classification report:")
print(c1)
```

## Output:

Dataset Info:

![31fa41a1-b885-4f08-93b8-aad388908093](https://github.com/user-attachments/assets/eb6e7167-8c0e-4a74-a762-f7ef48590783)

Y prediction:

![eb1c7fb2-a441-4248-94f1-0047c51f0c3e](https://github.com/user-attachments/assets/8a4a54ce-a36a-48ca-9ea2-bb987c905a09)

Accuracy:

![fe6e08ea-8e3f-48bc-94db-30e4753ece2a](https://github.com/user-attachments/assets/d4b11254-ed3f-426c-b36c-190871496924)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
