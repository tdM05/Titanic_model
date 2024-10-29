
from train_model import mae_model, generate_csv

features = ['Pclass', 'Age', 'Sex_indicator', 'Parch']
mae_model(features)
# generate_csv(features)