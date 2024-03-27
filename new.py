import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with 'msg_tx' and 'outage_indicator' columns
# Create a DataFrame (Replace this with your actual data loading process)
data = {
    'msg_tx': ['text data 1', 'text data 2', 'text data 3', 'text data 4'],
    'outage_indicator': ['indicator 1', 'indicator 2', 'indicator 1', 'indicator 2']
}
df = pd.DataFrame(data)

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

# Initialize the Bernoulli Naive Bayes classifier
bnb_classifier = BernoulliNB()

# Train the Bernoulli Naive Bayes classifier
bnb_classifier.fit(X_train_rf, y_train_rf)

# Predict on the testing data
bnb_predicted = bnb_classifier.predict(X_test_rf)

# Calculate accuracy
bnb_accuracy = accuracy_score(y_test_rf, bnb_predicted)
print(f'Bernoulli Naive Bayes Accuracy: {bnb_accuracy:.4f}')

# Classification report
print("Bernoulli Naive Bayes Classification Report:")
print(classification_report(y_test_rf, bnb_predicted))

# Confusion matrix
bnb_cm = confusion_matrix(y_test_rf, bnb_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(bnb_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Bernoulli Naive Bayes Confusion Matrix')
plt.show()
