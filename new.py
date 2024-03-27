from sklearn.linear_model import SGDClassifier

# Initialize the SGD classifier
sgd_classifier = SGDClassifier()

# Train the SGD classifier
sgd_classifier.fit(X_train_rf, y_train_rf)

# Predict on the testing data
sgd_predicted = sgd_classifier.predict(X_test_rf)

# Calculate accuracy
sgd_accuracy = accuracy_score(y_test_rf, sgd_predicted)
print(f'SGD Classifier Accuracy: {sgd_accuracy:.4f}')

# Classification report
print("SGD Classifier Classification Report:")
print(classification_report(y_test_rf, sgd_predicted))

# Confusion matrix
sgd_cm = confusion_matrix(y_test_rf, sgd_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(sgd_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('SGD Classifier Confusion Matrix')
plt.show()
