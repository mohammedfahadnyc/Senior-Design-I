def predict_labels(strings, trained_models, count_vectorizer):
    """
    Predict labels for a list of strings using trained models.

    Args:
    strings (list): List of strings to predict labels for.
    trained_models (dict): Dictionary containing trained models.
    count_vectorizer: CountVectorizer used for feature extraction.

    Returns:
    None
    """
    for string in strings:
        # Transform the string using the CountVectorizer
        string_count_matrix = count_vectorizer.transform([string])
        string_count_matrix_dense = string_count_matrix.toarray()

        # Predict labels using each trained model
        for name, model in trained_models.items():
            # Predict label
            predicted_label = model.predict(string_count_matrix_dense)[0]
            print(f"String: {string} | Predicted Label ({name}): {predicted_label}")

# Example usage:
strings_to_predict = ["example string 1", "example string 2", "example string 3"]
predict_labels(strings_to_predict, trained_models, count_vectorizer)
