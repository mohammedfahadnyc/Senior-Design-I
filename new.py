import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset (assuming 'df' is your DataFrame)
# Example: df = pd.read_csv('your_dataset.csv')

# Preprocessing: Assuming 'msg_tx' and 'outage_indicator' are the columns in the DataFrame
text_data = df['msg_tx']
labels = df['outage_indicator']

# Feature Engineering: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Adjust max_features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
tfidf_matrix_dense = tfidf_matrix.toarray()

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix_dense, encoded_labels, test_size=0.2, random_state=42)

# Display shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
