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







import org.apache.log4j.AppenderSkeleton
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.log4j.spi.LoggingEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RequestLoggerTest {

    @Test
    fun testCreateRequestLogger() {
        val appender = TestAppender()
        val logger = Logger.getRootLogger()
        logger.addAppender(appender)

        val createRequest = CreateRequest("John Doe", listOf("Jane Smith"))
        createRequestLogger(123, createRequest)

        val loggedMessages = appender.getLog()

        assertTrue(loggedMessages.isNotEmpty())

        val firstLogEntry = loggedMessages.first()
        assertEquals(Level.INFO, firstLogEntry.level)
        assertEquals("Successfully created GoLink with id 123, Owner is John Doe, Co-owner(s): Jane Smith.", firstLogEntry.message)
    }

    @Test
    fun testUpdateRequestLogger() {
        val appender = TestAppender()
        val logger = Logger.getRootLogger()
        logger.addAppender(appender)

        val updateRequest = UpdateRequest("Jane Smith")
        val prevGolink = Golink("John Doe")
        updateRequestLogger(updateRequest, prevGolink)

        val loggedMessages = appender.getLog()

        assertTrue(loggedMessages.isNotEmpty())

        val firstLogEntry = loggedMessages.first()
        assertEquals(Level.INFO, firstLogEntry.level)
        assertEquals("Updated GoLink owner from John Doe to Jane Smith.", firstLogEntry.message)
    }
}

class TestAppender : AppenderSkeleton() {
    private val log = mutableListOf<LoggingEvent>()

    override fun requiresLayout(): Boolean {
        return false
    }

    override fun append(loggingEvent: LoggingEvent) {
        log.add(loggingEvent)
    }

    override fun close() {}

    fun getLog(): List<LoggingEvent> {
        return log.toList()
    }
}

Updated\sGoLink\s(?:with\s(?:id|owner|co-owner|name|url):\s\d+|name\sfrom\s\w+\sto\s\w+)
val regex = Regex("""Updated\sGoLink\s(?:with\s(?:id|owner|co-owner|name|url):\s\d+|name\sfrom\s\w+\sto\s\w+)""")






