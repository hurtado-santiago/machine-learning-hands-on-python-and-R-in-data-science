# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# # Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])


# # Encoding categorical data
# # Encoding the Independent  Variable
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# x = onehotencoder.fit_transform(x).toarray()
# # Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_x.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.transform(x_test)
