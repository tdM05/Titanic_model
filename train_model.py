import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def mae_model(features: list[str], max_leaf_nodes: int) -> None:
    """Prints the mae"""
    train_path = "./resources/train.csv"
    # Load data
    data = pd.read_csv(train_path)
    # convert sex feature into numerical values
    data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked_indicator'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = data[features]
    y = data.Survived

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Define model
    model = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        random_state=1
    )
    model.fit(train_X, train_y)
    # print(type(data.PassengerId))
    # Get predictions
    predictions = model.predict(val_X)
    mae = mean_absolute_error(predictions, val_y)
    # print("Mean Absolute Error: ", mae)
    return mae
    # output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': predictions})
    # output.to_csv('splitted.csv', index=False)

def generate_csv(features: list[str]):
    """Prints the mae"""
    train_path = "./resources/train.csv"

    # Load data
    data = pd.read_csv(train_path)
    # convert sex feature into numerical values
    data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked_indicator'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = data[features]
    y = data.Survived
    # Define model
    model = DecisionTreeClassifier(random_state=1)
    model.fit(X, y)

    # Get predictions
    test_path = "./resources/test.csv"
    test_data = pd.read_csv(test_path)
    test_data['Sex_indicator'] = test_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Embarked_indicator'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    predictions = model.predict(test_data[features])
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)

def mae_forest_model(features: list[str], max_leaf_nodes: int) -> None:
    """Prints the mae"""
    train_path = "./resources/train.csv"
    # Load data
    data = pd.read_csv(train_path)
    # convert sex feature into numerical values
    data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked_indicator'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = data[features]
    y = data.Survived

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Define model
    model = RandomForestClassifier(
        max_leaf_nodes=max_leaf_nodes,
        random_state=1
    )
    model.fit(train_X, train_y)
    # print(type(data.PassengerId))
    # Get predictions
    predictions = model.predict(val_X)
    mae = mean_absolute_error(predictions, val_y)
    # print("Mean Absolute Error: ", mae)
    return mae
    # output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': predictions})
    # output.to_csv('splitted.csv', index=False)

def generate_forest_csv(features: list[str], max_leaf_nodes: int):
    """Prints the mae"""
    train_path = "./resources/train.csv"

    # Load data
    data = pd.read_csv(train_path)
    # convert sex feature into numerical values
    data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked_indicator'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = data[features]
    y = data.Survived
    # Define model
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(X, y)

    # Get predictions
    test_path = "./resources/test.csv"
    test_data = pd.read_csv(test_path)
    test_data['Sex_indicator'] = test_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Embarked_indicator'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    predictions = model.predict(test_data[features])
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)