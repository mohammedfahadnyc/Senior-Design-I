from sklearn.naive_bayes import MultinomialNB

# Initialize the Multinomial Naive Bayes classifier
mnb_classifier = MultinomialNB()

# Train the Multinomial Naive Bayes classifier
mnb_classifier.fit(X_train_rf, y_train_rf)

# Predict on the testing data
mnb_predicted = mnb_classifier.predict(X_test_rf)

# Calculate accuracy
mnb_accuracy = accuracy_score(y_test_rf, mnb_predicted)
print(f'Multinomial Naive Bayes Accuracy: {mnb_accuracy:.4f}')

# Classification report
print("Multinomial Naive Bayes Classification Report:")
print(classification_report(y_test_rf, mnb_predicted))

# Confusion matrix
mnb_cm = confusion_matrix(y_test_rf, mnb_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(mnb_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Multinomial Naive Bayes Confusion Matrix')
plt.show()
