import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the new dataset
new_df = pd.read_csv('b.csv')  # Replace 'b.csv' with the actual filename/path

# Preprocessing: Assuming 'msg_tx' is the text data
new_text_data = new_df['msg_tx']
new_labels = new_df['outage_indicator']

# Feature Engineering: CountVectorizer
count_vectorizer = CountVectorizer(max_features=10000)  # Adjust max_features as needed
count_matrix_new = count_vectorizer.fit_transform(new_text_data)
count_matrix_dense_new = count_matrix_new.toarray()

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels_new = label_encoder.fit_transform(new_labels)

# Train-Test Split for new data
X_train_rf_new, X_test_rf_new, y_train_rf_new, y_test_rf_new = train_test_split(count_matrix_dense_new, encoded_labels_new, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier for new data
rf_classifier_new = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier on new data
rf_classifier_new.fit(X_train_rf_new, y_train_rf_new)

# Predict on the testing data for new data
rf_predicted_new = rf_classifier_new.predict(X_test_rf_new)

# Calculate accuracy for new data
rf_accuracy_new = accuracy_score(y_test_rf_new, rf_predicted_new)
print(f'New Random Forest Accuracy: {rf_accuracy_new:.4f}')

# Classification report for new data
print("New Random Forest Classification Report:")
print(classification_report(y_test_rf_new, rf_predicted_new))

# Confusion matrix for new data
rf_cm_new = confusion_matrix(y_test_rf_new, rf_predicted_new)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm_new, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('New Random Forest Confusion Matrix')
plt.show()
