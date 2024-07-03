# Stroke-Predictor-Model : Machine Learning Model for Forecasting Stroce with nearly 98% accuracy

## Introduction

In today's healthcare landscape, predictive modeling plays a crucial role in preemptively identifying health risks and guiding preventive care strategies. This project demonstrates the application of machine learning techniques to predict strokes using the Healthcare Dataset Stroke available on Kaggle. By analyzing factors such as age, hypertension, heart disease, and glucose levels, this model aims to assist healthcare professionals in early identification and intervention for stroke-prone individuals.


## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Predictive Modeling](#predictive-modeling)
6. [Model Evaluation](#model-evaluation)
7. [Actionable Marketing Insights](#actionable-marketing-insights)
8. [Implementation and Deployment](#implementation-and-deployment)
9. [Conclusion](#conclusion)

# Data Preparation

### Importing Necessary Libraries

First, import the necessary libraries for data analysis and machine learning.

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lazypredict
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE
```

## Load dataset using full path
Load the dataset Load the dataset into DataFrame.

```python
df = pd.read_csv("/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv")
```

## Display the first few rows of the dataset
inspect the first few rows to understand its structure.
```python
df.head()
```

![image]([https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/ba4ade09-2909-4b12-bf48-071857926230](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/a8d53b28-771e-454a-9506-8dca4a559b97))


## How Many Instances and Features ?
Check for and handle missing values to ensure a clean dataset.
```python
df.shape
```

## Checking for missing values
Check for and handle missing values to ensure a clean dataset.

```python
df.isna().sum()
```

![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/7faa8c62-6897-4b7f-a005-be7e2d54751b)


## Replace rows with missing values (with mean since it is numerical column)

```python
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
```
##Drop 'id' Column because it will not help in predicting
```python
df = df.drop("id",axis=1)
```

## Didscover if Target column values are Balanced or not 
Notice it is not balanced so we will work on this problem later

```python
df['stroke'].value_counts()
```
![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/74c9f0a9-e9d0-4985-b976-a1b6b5bcd04f)

##Display Variables data type and number of non NULLs values in

```python
df.info()
```


# Data Transformation

Convert categorical variables into numerical ones for machine learning models.

```python
df = pd.get_dummies(df).astype(int)
```

## Summary statistics

```python
print(df.describe())
```

![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/9c1634ee-62c9-4b5e-a85e-8ce1fee70ca4)

## Key Observations
- Age: The average age is 43, first quarter is to age of 25 and maximum age is 82, wide range indicates a diverse base in terms of age.
- glucose_level: On average, glucose_level about 105 , with most ranging from 55 to 115.
- Body Mass Index 'bmi': On average, Body Mass Index about 28 , ranging between 10 and 97 with most ranging from 10 to 32.

## Calculate the correlation matrix

```python
corr=df.corr()
```

## Create a correlation heatmap for the subset of features

```python
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,mask = np.triu(np.ones_like(corr, dtype=bool)))
```
![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/cb4c0184-4189-4e59-b3bb-9de6fb29041a)


## Split data into training and testing sets

```python
x = df.drop("stroke",axis=1)
y = df.stroke
```

## Import SMOTE
smote is a library used to Handle the imbalanced data by Reducing the size of major class to be as ther minor or by increase the size of minor class
in our case we will increase the minor class instances
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy="minority")
x_smote , y_smote = smote.fit_resample(x,y)       
```

## Import LazyPredict
it is a library that is used to evaluate many algorithms and find the accuracies for each one
in this case we find that RandomForestClassifier achieves the best accuracy 
```python
import lazypredict
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)

```


## Evaluate models
after training the model and predicting it on test data it makes accuracy of nearly 98%
![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/9e02981f-6a88-460b-9098-5b69057aa10d)

##Creating Confusion Matrix
it shows the predicted values Distribution

```python
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
```
![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/c721c5ea-f45a-4415-8fcb-28747c7a4c35)


