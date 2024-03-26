import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Assuming you have already preprocessed your text data and loaded it into a DataFrame called df
# df should contain 'msg_tx' column for messages and 'outage_indicator' column for labels

# Step 1: Tokenize the text data and encode the labels
# Tokenization
messages = df['msg_tx'].values

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(df['outage_indicator'].values)

# Step 2: Split the dataset into training and testing sets
messages_train, messages_test, labels_train, labels_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Step 3: Convert text data into numerical features
# Convert text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(messages_train)

# Convert raw frequency counts into TF-IDF (Term Frequency-Inverse Document Frequency) values
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Step 4: Padding sequences (assuming you're using LSTM)
# Find the maximum length of messages
max_len = max(len(message.split()) for message in messages_train)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages_train)
sequences_train = tokenizer.texts_to_sequences(messages_train)
sequences_test = tokenizer.texts_to_sequences(messages_test)

X_train = pad_sequences(sequences_train, maxlen=max_len)
X_test = pad_sequences(sequences_test, maxlen=max_len)

# Step 5: Convert labels into categorical format
y_train = to_categorical(labels_train)
y_test = to_categorical(labels_test)

# Now, you can proceed to train your LSTM model using X_train, y_train, X_test, and y_test
# Make sure to define your LSTM model architecture and train it accordingly
