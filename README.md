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

