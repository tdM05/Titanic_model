
from train_model import mae_model, generate_csv

features1 = ['Pclass', 'Age', 'Sex_indicator', 'Parch']
features2 = ['Pclass', 'Age', 'Sex_indicator', 'Parch',
             'SibSp', 'Fare', 'Embarked_indicator']
mae_model(features2, 4)
generate_csv(features2)
# features_list= [
#
# ]
# min_mae = -1
# min_nodes = -1
# for i in range(2, 100):
#     mae = mae_model(features, i)
#     if min_mae == -1 or min_mae < mae:
#         min_mae = mae
#         min_nodes = i
# print(f"min_nodes:{min_nodes}, min_mae:{min_mae}")

# generate_csv(features, 4)