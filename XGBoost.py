import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load the data
cars_file_path = '/Users/maya/Downloads/ToyotaCorolla.csv'
cars_data = pd.read_csv(cars_file_path)

# Drop rows with missing values
cars_data = cars_data.dropna(axis=0)

# Define target variable and features
y = cars_data['Price']

# Select subset of predictors
cols_to_use = ['Age_08_04', 'KM', 'HP', 'CC', 'Weight']
X = cars_data[cols_to_use]

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

#Defining model
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

#Accuracy score
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))