from sklearn.linear_model import PassiveAggressiveClassifier

# Initialize the Passive Aggressive classifier
pa_classifier = PassiveAggressiveClassifier()

# Train the Passive Aggressive classifier
pa_classifier.fit(X_train_rf, y_train_rf)

# Predict on the testing data
pa_predicted = pa_classifier.predict(X_test_rf)

# Calculate accuracy
pa_accuracy = accuracy_score(y_test_rf, pa_predicted)
print(f'Passive Aggressive Classifier Accuracy: {pa_accuracy:.4f}')

# Classification report
print("Passive Aggressive Classifier Classification Report:")
print(classification_report(y_test_rf, pa_predicted))

# Confusion matrix
pa_cm = confusion_matrix(y_test_rf, pa_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(pa_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Passive Aggressive Classifier Confusion Matrix')
plt.show()
