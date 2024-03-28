
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier

# Load your actual data (Replace this with your actual data loading process)
df = pd.read_csv("your_data.csv")  # Adjust filename as needed

# Preprocessing: Assuming 'msg_tx' is the text data and 'outage_indicator' is the target variable
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
    "Perceptron": Perceptron(),
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

# Display results
for name, (accuracy, report) in results.items():
    print(f"Model: {name}")
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("\n")
