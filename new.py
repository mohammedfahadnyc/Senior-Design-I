import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
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
X_train, X_test, y_train, y_test = train_test_split(count_matrix_dense, encoded_labels, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Perceptron": Perceptron(),
    "SGD Classifier": SGDClassifier(),
    "Passive Aggressive Classifier": PassiveAggressiveClassifier()
}

# Train classifiers
trained_models = {}
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    trained_models[name] = classifier

# Function to evaluate and return results as string
def evaluate_model(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted)
    return accuracy, report

# Evaluate trained models
results = {}
for name, model in trained_models.items():
    accuracy, report = evaluate_model(model, X_test, y_test)
    results[name] = (accuracy, report)

# Load updated data
updated_df = pd.read_csv("updated.csv")  # Adjust filename as needed

# Preprocess updated data
updated_text_data = updated_df['msg_tx']
updated_labels = updated_df['outage_indicator']
updated_count_matrix = count_vectorizer.transform(updated_text_data)

# Partial fit on updated data
for name, model in trained_models.items():
    model.partial_fit(updated_count_matrix, updated_labels)

# Evaluate updated models
updated_results = {}
for name, model in trained_models.items():
    updated_accuracy, updated_report = evaluate_model(model, updated_count_matrix, updated_labels)
    updated_results[name] = (updated_accuracy, updated_report)

# Display results side by side
for name in results.keys():
    print(f"Model: {name}")
    print("Original Accuracy:", results[name][0])
    print("Original Classification Report:")
    print(results[name][1])
    print("Updated Accuracy:", updated_results[name][0])
    print("Updated Classification Report:")
    print(updated_results[name][1])
    print("\n")
