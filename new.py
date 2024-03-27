from sklearn.linear_model import Perceptron

# Initialize the Perceptron classifier
perceptron_classifier = Perceptron()

# Train the Perceptron classifier
perceptron_classifier.fit(X_train_rf, y_train_rf)

# Predict on the testing data
perceptron_predicted = perceptron_classifier.predict(X_test_rf)

# Calculate accuracy
perceptron_accuracy = accuracy_score(y_test_rf, perceptron_predicted)
print(f'Perceptron Accuracy: {perceptron_accuracy:.4f}')

# Classification report
print("Perceptron Classification Report:")
print(classification_report(y_test_rf, perceptron_predicted))

# Confusion matrix
perceptron_cm = confusion_matrix(y_test_rf, perceptron_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(perceptron_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Perceptron Confusion Matrix')
plt.show()
