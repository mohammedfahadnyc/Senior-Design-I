import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the new dataset
new_df = pd.read_csv('b.csv')

# Preprocess the new dataset
new_text_data = new_df['msg_tx']
new_labels = new_df['outage_indicator']
new_count_matrix = count_vectorizer.transform(new_text_data)  # Reusing count_vectorizer from previous training
new_count_matrix_dense = new_count_matrix.toarray()
new_encoded_labels = label_encoder.transform(new_labels)  # Reusing label_encoder from previous training

# Initialize the RandomForestClassifier (assuming rf_classifier is the existing model)
rf_classifier = rf_classifier

# Train the model on the new dataset using partial_fit
rf_classifier.partial_fit(new_count_matrix_dense, new_encoded_labels, classes=label_encoder.classes_)

# Optionally, you can evaluate the performance of the updated model on a separate test set
# For example:
new_X_test_rf = new_count_vectorizer.transform(new_df['msg_tx']).toarray()
new_y_test_rf = label_encoder.transform(new_df['outage_indicator'])
new_rf_predicted = rf_classifier.predict(new_X_test_rf)
new_rf_accuracy = accuracy_score(new_y_test_rf, new_rf_predicted)
print(f'Updated Random Forest Accuracy: {new_rf_accuracy:.4f}')
