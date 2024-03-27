from sklearn.feature_extraction.text import CountVectorizer

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
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(count_matrix_dense, encoded_labels, test_size=0.2, random_state=42)

# Display shapes of the training and testing sets for SVM
print("Shape of X_train_svm:", X_train_svm.shape)
print("Shape of X_test_svm:", X_test_svm.shape)
print("Shape of y_train_svm:", y_train_svm.shape)
print("Shape of y_test_svm:", y_test_svm.shape)
