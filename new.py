
import pickle

# Assuming rf_model, perceptron_model, and pa_model are your trained models

# Save the Random Forest model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save the Perceptron model
with open("perceptron_model.pkl", "wb") as f:
    pickle.dump(perceptron_model, f)

# Save the Passive Aggressive model
with open("passive_aggressive_model.pkl", "wb") as f:
    pickle.dump(pa_model, f)



import pandas as pd
from sklearn.externals import joblib

# Load the trained models
rf_model = joblib.load("random_forest_model.pkl")  # Replace "random_forest_model.pkl" with the path to your Random Forest model
perceptron_model = joblib.load("perceptron_model.pkl")  # Replace "perceptron_model.pkl" with the path to your Perceptron model
pa_model = joblib.load("passive_aggressive_model.pkl")  # Replace "passive_aggressive_model.pkl" with the path to your Passive Aggressive model

# Load the 15 day dataset from Excel
train_data_15_days = pd.read_excel("train_data_15_days.xlsx")  # Adjust filename as needed

# Assuming 'msg_tx' is the text data in your 15 day dataset
# Predict pseudo labels using each model
train_data_15_days['pseudo_label_rf'] = rf_model.predict(train_data_15_days['msg_tx'])
train_data_15_days['pseudo_label_perceptron'] = perceptron_model.predict(train_data_15_days['msg_tx'])
train_data_15_days['pseudo_label_pa'] = pa_model.predict(train_data_15_days['msg_tx'])

# Now train_data_15_days contains pseudo labels assigned by each model in separate columns
