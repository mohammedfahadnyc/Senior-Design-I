import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example data
texts = df['msg_tx'].tolist()  # Assuming 'df' is your DataFrame containing the messages
labels = df['outage_indicator'].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert text data to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
max_seq_length = 100  # Choose a suitable maximum sequence length based on your data
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# Load pre-trained word embeddings (e.g., GloVe embeddings)
# Download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/
embedding_dim = 100  # Adjust the embedding dimension based on the chosen pre-trained embeddings
embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create embedding matrix
word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Alternatively, you can use pre-trained embeddings provided by Keras
# from tensorflow.keras.layers import Embedding
# embedding_layer = Embedding(num_words, embedding_dim, embeddings_initializer='glorot_uniform',
#                             input_length=max_seq_length, trainable=False)

# Now, 'padded_sequences' contains the tokenized and padded sequences,
# and 'embedding_matrix' contains the corresponding word embeddings.
# You can use these as input features for training your LSTM model.
