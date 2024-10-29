import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def mae_model(features: list[str]) -> None:
    """Prints the mae"""
    train_path = "./resources/train.csv"
    # Load data
    data = pd.read_csv(train_path)
    # convert sex feature into numerical values
    data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
    X = data[features]
    y = data.Survived

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Define model
    model = DecisionTreeRegressor(random_state=1)
    model.fit(train_X, train_y)

    # Get predictions
    predictions = model.predict(val_X)
    mae = mean_absolute_error(predictions, val_y)
    print("Mean Absolute Error: ", mae)

    # output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': predictions})
    # output.to_csv('splitted.csv', index=False)

def generate_csv(features: list[str]):
    """Prints the mae"""
    train_path = "./resources/train.csv"

    # Load data
    data = pd.read_csv(train_path)
    # convert sex feature into numerical values
    data['Sex_indicator'] = data['Sex'].map({'male': 0, 'female': 1})
    X = data[features]
    y = data.Survived
    # Define model
    model = DecisionTreeRegressor(random_state=1)
    model.fit(X, y)

    # Get predictions
    test_path = "./resources/test.csv"
    test_data = pd.read_csv(test_path)
    test_data['Sex_indicator'] = test_data['Sex'].map({'male': 0, 'female': 1})
    predictions = model.predict(test_data[features])
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)

