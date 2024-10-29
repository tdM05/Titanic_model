import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

train_path = "./resources/train.csv"

# Load data
data = pd.read_csv(train_path)
# print(data.columns)
# print(data.describe())
y = data.Survived

# convert sex feature into numerical values
data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
print(data.columns)