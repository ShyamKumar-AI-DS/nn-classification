# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### Step 1:
We start by reading the dataset using pandas.

### Step 2:
The dataset is then preprocessed, i.e, we remove the features that don't contribute towards the result.

### Step 3:
The null values are removed aswell

### Step 4:
The resulting data values are then encoded. We, ensure that all the features are of the type int, or float, for the model to better process the dataset.

### Step 5:
Once the preprocessing is done, we split the available data into Training and Validation datasets.

### Step 6:
The Sequential model is then build using 1 input, 3 dense layers(hidden) and, output layer.

### Step 7:
The model is then complied and trained with the data. A call back method is also implemented to prevent the model from overfitting.

### Step 8:
Once the model is done training, we validate and use the model to predict values.

## PROGRAM

~~~
Name : Shyam Kumar A 
Reg No : 212221230098
~~~
~~~
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
import pickle
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pylab as plt

df = pd.read_csv("customers.csv")

df.columns

df.dtypes

df.shape

df.isnull().sum()

df1 = df.dropna(axis = 0)

df1.isnull().sum()

df1.shape

df1.dtypes

df1['Gender'].unique(),df1['Ever_Married'].unique(),df1['Graduated'].unique(),df1['Profession'].unique(),df1['Spending_Score'].unique(),df1['Var_1'].unique(),df1['Segmentation'].unique()

category_list = [['Male', 'Female'],['No', 'Yes'],['No', 'Yes'],['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
        'Homemaker', 'Entertainment', 'Marketing', 'Executive'],['Low', 'High', 'Average']]

enc = OrdinalEncoder(categories = category_list)

customer1 = df1.copy()

customer1[['Gender','Ever_Married'
          ,'Graduated','Profession'
            ,'Spending_Score']] = enc.fit_transform(customer1[['Gender','Ever_Married'
                                                             ,'Graduated','Profession'
                                                             ,'Spending_Score']])

customer1.dtypes

Le = LabelEncoder()

customer1['Segmentation'] = Le.fit_transform(customer1['Segmentation'])

customer1 = customer1.drop('ID',axis=1)
customer1 = customer1.drop('Var_1',axis=1)

customer1.dtypes

corr = customer1.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)

X=customer1[['Gender','Age','Ever_Married','Graduated','Profession','Spending_Score','Work_Experience','Family_Size']].values

y1 = customer1[['Segmentation']].values

onehot_enc = OneHotEncoder()

onehot_enc.fit(y1)

y = onehot_enc.transform(y1).toarray()

y.shape

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33,random_state=50)

xtrain[0]

xtrain.shape

xtest.shape

scaler_age = MinMaxScaler()

scaler_age.fit(xtrain[:,2].reshape(-1,1))

xtrain_scaled = np.copy(xtrain)
xtest_scaled = np.copy(xtest)

xtrain_scaled[:,2] = scaler_age.transform(xtrain[:,2].reshape(-1,1)).reshape(-1)
xtest_scaled[:,2] = scaler_age.transform(xtest[:,2].reshape(-1,1)).reshape(-1)

 ai_brain = Sequential([
    Dense(8,input_shape=(8,)),
    Dense(10,activation='relu'),
    Dense(12,activation='relu'),
    Dense(16,activation='relu'),
    Dense(32,activation='relu'),
    Dense(64,activation='relu'),
    Dense(128,activation='relu'),
    Dense(4,activation='softmax')
 ])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)

ai_brain.fit(x=xtrain_scaled,y=ytrain,
             epochs=2000,batch_size=256,
             validation_data=(xtest_scaled,ytest),
             )

metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

xtest_predictions = np.argmax(ai_brain.predict(xtest_scaled), axis=1)

xtest_predictions.shape

ytest_truevalue = np.argmax(ytest,axis=1)

ytest_truevalue.shape

print(confusion_matrix(ytest_truevalue,xtest_predictions))

print(classification_report(ytest_truevalue,xtest_predictions))

x_single_prediction = np.argmax(ai_brain.predict(xtest_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(Le.inverse_transform(x_single_prediction))

~~~

## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here


### New Sample Data Prediction

Include your sample input and output here

## RESULT
