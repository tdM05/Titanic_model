
from train_model import mae_model, generate_csv, generate_forest_csv, mae_forest_model

features1 = ['Pclass', 'Age', 'Sex_indicator', 'Parch']
features2 = ['Pclass', 'Age', 'Sex_indicator', 'Parch',
             'SibSp', 'Fare', 'Embarked_indicator']
features3 = ['Pclass', 'Age', 'Sex_indicator', 'Parch',
             'SibSp', 'Fare']
features4 = ['Pclass', 'Age', 'Sex_indicator', 'Parch',
             'SibSp']
features5 = ['Pclass', 'Age', 'Sex_indicator', 'Parch',
             'Fare']
features_list = [features1, features2, features3, features4, features5]
print(mae_forest_model(features3, 6))

# for features in features_list:
#     min_mae = -1
#     min_nodes = -1
#     for i in range(2, 100):
#         mae = mae_model(features, i)
#         if min_mae == -1 or min_mae > mae:
#             min_mae = mae
#             min_nodes = i
#     print(f"min_nodes:{min_nodes}, min_mae:{min_mae}")

# min_nodes 6 is optimal for features 3
# generate_csv(features, 4)