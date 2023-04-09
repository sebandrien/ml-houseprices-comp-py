import warnings #Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
 
df = pd.read_csv("housing.csv") #Read in "housing" to "df"

df.head() #Display "df's" head

df.tail() #Display "df's" tail

df.info() #Display "df's" info

df.describe() #Describe "df's" data

transposed_df = df.transpose().isna() #Visualize negative values
plt.figure(dpi=300)
fig, ax = plt.subplots(figsize = (20,10))
sns.heatmap(transposed_df, ax = ax, cmap = "viridis")
plt.show()

plt.figure(dpi=300) #Display histogram of "df"
df.hist(figsize = (20,10))
plt.show()

df = pd.get_dummies(df, columns = ["ocean_proximity"], drop_first = True) #"ocean_proximity" is an object so it needs to be convereted to a float value

df = df.dropna(axis = 0) #Drop null valus

df.corr()["median_house_value"].sort_values(ascending = False) #View correlation

y = df[["median_house_value"]] #
X = df.drop(["median_house_value"], axis = 1) #Drop 

RS = 123
TS = 0.25
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = TS, random_state = RS)c
