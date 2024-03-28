from sklearn.ensemble import RandomForestClassifier

# Assuming count_vectorizer is your CountVectorizer object from previous preprocessing
# Preprocess the text data in the 15-day dataset
text_data_15_days = train_data_15_days['msg_tx']
count_matrix_15_days = count_vectorizer.transform(text_data_15_days)

# Initialize the Random Forest classifier with the same parameters as before
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train_rf, y_train_rf)

# Predict on the 15-day data
pseudo_labels_rf = rf_classifier.predict(count_matrix_15_days)

# Add pseudo labels to the 15-day dataset
train_data_15_days['pseudo_label_rf'] = pseudo_labels_rf

# Now train_data_15_days contains pseudo labels assigned by the Random Forest classifier
# Adjust the filenames, column names, and preprocessing steps as needed based on your actual data.

