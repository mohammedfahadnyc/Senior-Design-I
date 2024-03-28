from sklearn.ensemble import RandomForestClassifier

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

# Load the 15 day dataset from Excel
train_data_15_days = pd.read_excel("train_data_15_days.xlsx")  # Adjust filename as needed

# Preprocess the text data in the 15-day dataset
text_data_15_days = train_data_15_days['msg_tx']
count_matrix_15_days = count_vectorizer.transform(text_data_15_days)
count_matrix_dense_15_days = count_matrix_15_days.toarray()

# Assign pseudo labels using the trained Random Forest classifier
train_data_15_days['pseudo_label_rf'] = rf_classifier.predict(count_matrix_dense_15_days)

# Now train_data_15_days contains pseudo labels assigned by the Random Forest classifier in a new column 'pseudo_label_rf'
# You can save train_data_15_days to a new Excel file or further process it as needed
