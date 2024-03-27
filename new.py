from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Preprocessing: Assuming 'msg_tx' is the text data
text_data = df['msg_tx']
labels = df['outage_indicator']

# Feature Engineering: CountVectorizer
count_vectorizer = CountVectorizer(max_features=10000)  # Adjust max_features as needed
count_matrix = count_vectorizer.fit_transform(text_data)
count_matrix_dense = count_matrix.toarray()

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train-Test Split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(count_matrix_dense, encoded_labels, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train_rf, y_train_rf)

# Predict on the testing data
rf_predicted = rf_classifier.predict(X_test_rf)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test_rf, rf_predicted)
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

# Classification report
print("Random Forest Classification Report:")
print(classification_report(y_test_rf, rf_predicted))

# Confusion matrix
rf_cm = confusion_matrix(y_test_rf, rf_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix')
plt.show()
