from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM model
svm_model = SVC(kernel='linear', C=1.0)

# Flatten the input tensors for SVM
X_train_flatten = X_train_tensor.view(X_train_tensor.size(0), -1)
X_test_flatten = X_test_tensor.view(X_test_tensor.size(0), -1)

# Train the SVM model
svm_model.fit(X_train_flatten.numpy(), y_train_tensor.numpy())

# Predict on the testing data
svm_predicted = svm_model.predict(X_test_flatten.numpy())

# Calculate accuracy
svm_accuracy = accuracy_score(y_test_tensor.numpy(), svm_predicted)
print(f'SVM Accuracy: {svm_accuracy:.4f}')
