import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained models
perceptron_model = Perceptron()
pa_model = PassiveAggressiveClassifier()

# Load the 15 day dataset from Excel
train_data_15_days = pd.read_excel("train_data_15_days.xlsx")  # Adjust filename as needed

# Load the CountVectorizer from the Perceptron model
count_vectorizer = perceptron_model.named_steps['vectorizer']

# Preprocess the text data
text_data = train_data_15_days['msg_tx']

# Transform the text data using the loaded CountVectorizer
count_matrix_dense = count_vectorizer.transform(text_data).toarray()

# Assign pseudo labels using each model
train_data_15_days['pseudo_label_perceptron'] = perceptron_model.predict(count_matrix_dense)
train_data_15_days['pseudo_label_pa'] = pa_model.predict(count_matrix_dense)

# Now train_data_15_days contains pseudo labels assigned by each model in separate columns
