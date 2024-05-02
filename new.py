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








fun updateRequestLogger(updateRequest: UpdateRequest, prevGolink: Golink?){
        val previousCreatedBy: String? =
            if (!updateRequest.createdBy.isNullOrEmpty()) {
                prevGolink?.createdBy
            } else {
                ""
            }
        val previousCoOwner : String?=
            if((!updateRequest.coOwner.isNullOrEmpty())) {
                prevGolink?.coOwner
            }
            else {
                ""
            }
        val previousName : String?=
            if((!updateRequest.name.isNullOrEmpty())) {
                prevGolink?.name
            }
            else {
                ""
            }
        val previousUrl : String?=
            if((!updateRequest.url.isNullOrEmpty())) {
                prevGolink?.url
            }
            else {
                ""
            }
        logger.info { "Updated GoLink with id: ${updateRequest.id}" }

        if (previousCreatedBy?.isNotEmpty() == true && previousCreatedBy != updateRequest.createdBy) {
            logger.info { "Updated GoLink owner from ${previousCreatedBy} to ${updateRequest.createdBy}." }
        }
        // Log when the co-owner of the golink has been added
        if (previousCoOwner?.isEmpty() == true  && previousCoOwner != updateRequest.coOwner) {
            logger.info { "Added GoLink co-owner : ${updateRequest.coOwner}." }
        }
        // Log when the co-owner of the golink has changed
        if (previousCoOwner?.isNotEmpty() == true && previousCoOwner != updateRequest.coOwner) {
            logger.info { "Updated GoLink co-owner from ${previousCoOwner} to ${updateRequest.coOwner}." }
        }
        // Log when the name of the golink has changed
        if (previousName?.isNotEmpty() == true  && previousName != updateRequest.name) {
            logger.info { "Updated GoLink name from ${previousName} to ${updateRequest.name}." }
        }
        // Log when the url of the golink has changed
        if (previousUrl?.isNotEmpty() == true && previousUrl != updateRequest.url) {
            logger.info { "Updated GoLink url from ${previousUrl} to ${updateRequest.url}." }
        }
    }

urls.forEach(url => {
    console.log(`${url} is a URL: ${isURL(url)}`);
});
