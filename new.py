import pickle

# Load the Random Forest model
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load the Perceptron model
with open("perceptron_model.pkl", "rb") as f:
    perceptron_model = pickle.load(f)

# Load the Passive Aggressive model
with open("passive_aggressive_model.pkl", "rb") as f:
    pa_model = pickle.load(f)
