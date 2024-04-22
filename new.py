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


const urlRegex = /^(https?:\/\/)?([\w-]+\.)*[\w-]+(\.[a-z]{2,})(:\d{1,5})?(\/\S*)?$/i;
const urlRegex = /^(?:(?:https?:\/\/)?|(?:www\.)?)[\w-]+(?:\.[\w-]+)+(?:\:\d{1,5})?(?:\/\S*)?$/i;

const urlRegex = /^(?:(?:https?|ftp):\/\/|slack:\/\/)[\w-]+(?:\.[\w-]+)+(?:\:\d{1,5})?(?:\/\S*)?$/i;




const urlRegex = /^(\w+:\/\/|\/)\S*$/i;




const urlRegex = /^(?:(?:https?|ftp):\/\/)?(?:\S+(?::\S*)?@)?(?:(?!-)[A-Z\d-]{1,63}(?<!-)\.?)+(?:[A-Z]{2,63}|(?:com|org|net|...)\b)(?:\/\S*)?$/i;
const urlRegex = /^(?:(?:https?|ftp):\/\/)?(?:\S+(?::\S*)?@)?(?:(?!-)[A-Z\d-]{1,63}(?<!-)\.?)+(?:[A-Z]{2,63}|(?:com|org|net|...)\b)(?:\/\S*)?$/i;

function isURL(str) {
    return urlRegex.test(str);
}

// Test cases
const urls = [
    "https://www.example.com",
    "http://subdomain.example.com",
    "ftp://example.com",
    "www.example.com",
    "example.com",
    "https://example.com:8080/path",
    "http://sub.example.com/path/to/page.html",
    "https://sub.example.com:8080/path/to/page.html?query=string",
    "not_a_url"
];

urls.forEach(url => {
    console.log(`${url} is a URL: ${isURL(url)}`);
});
