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

# Preprocessing: Assuming 'msg_tx' is the text data
text_data = train_data_15_days['msg_tx']

# Use the same CountVectorizer as before
count_vectorizer = CountVectorizer(max_features=10000)
count_matrix = count_vectorizer.fit_transform(text_data)
count_matrix_dense = count_matrix.toarray()

# Initialize label encoder
label_encoder = LabelEncoder()

# Assign pseudo labels using each model
train_data_15_days['pseudo_label_perceptron'] = perceptron_model.predict(count_matrix_dense)
train_data_15_days['pseudo_label_pa'] = pa_model.predict(count_matrix_dense)

# Now train_data_15_days contains pseudo labels assigned by each model in separate columns

