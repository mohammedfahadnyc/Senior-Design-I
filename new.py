# Load the new dataset
new_df = pd.read_csv('b.csv')  # Replace 'b.csv' with the actual filename/path

# Preprocessing: Assuming 'msg_tx' is the text data
new_text_data = new_df['msg_tx']
new_labels = new_df['outage_indicator']

# Feature Engineering: CountVectorizer
count_matrix_new = count_vectorizer.transform(new_text_data)
count_matrix_dense_new = count_matrix_new.toarray()

# Label Encoding
encoded_labels_new = label_encoder.transform(new_labels)

# Update the existing Random Forest classifier with new data
rf_classifier.partial_fit(count_matrix_dense_new, encoded_labels_new)

# Predict on the testing data for new data
rf_predicted_new = rf_classifier.predict(X_test_rf_new)

# Calculate accuracy for new data
rf_accuracy_new = accuracy_score(y_test_rf_new, rf_predicted_new)
print(f'Updated Random Forest Accuracy: {rf_accuracy_new:.4f}')

# Classification report for new data
print("Updated Random Forest Classification Report:")
print(classification_report(y_test_rf_new, rf_predicted_new))

# Confusion matrix for new data
rf_cm_new = confusion_matrix(y_test_rf_new, rf_predicted_new)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm_new, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Updated Random Forest Confusion Matrix')
plt.show()
