<H3>ENTER YOUR NAME : JAYAVARSHA T</H3>
<H3>ENTER YOUR REGISTER NO : 212223040075</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 28.08.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
Developed by: JAYAVARSHA T
RegisterNumber: 212223040075

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))


## OUTPUT:
<H3>DATASET:</H3>
![output 1](https://github.com/user-attachments/assets/82ddbdc4-a57d-467d-88e0-c9ee2aad7043)

<H3>DROPPING THE UNWANTED DATASET:</H3>
![NN output](https://github.com/user-attachments/assets/1360e887-50f4-410a-9907-329ece3f4411)

<H3>CHECKING NULL VALUES:</H3>
![ouput 2](https://github.com/user-attachments/assets/9dc96e34-4852-4fac-9ebc-1ad335033400)

<H3>CHECKING FOR DUPLICATION:</H3>
![output 3](https://github.com/user-attachments/assets/91424690-d941-47c9-aae9-ae00cde02345)

<H3>DESCRIBING THE DATASET:</H3>
![output 4](https://github.com/user-attachments/assets/05113bcd-54e6-4c48-9d64-13e6b2f05667)

<H3>SCALING THE DATASET:</H3>
![output 5](https://github.com/user-attachments/assets/5a5830c9-e5cb-42cf-854b-26c163330a57)

<H3>X FEATURES:</H3>
![output 6](https://github.com/user-attachments/assets/4c907d80-66b1-454d-afc3-12b94ccd652e)

<H3>Y FEATURES:</H3>
![output 7](https://github.com/user-attachments/assets/1ba355ed-d285-4185-8f1d-fe2fd2c817ad)

<H3>SPLITTING THE TRAINING AND TESTING DATASET:</H3>
![output 8](https://github.com/user-attachments/assets/3647524d-e2e3-4c8f-9d89-a365f0964d4d)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


