
# Load the 15 day dataset from Excel
train_data_15_days = pd.read_excel("train_data_15_days.xlsx")  # Adjust filename as needed

# Preprocess the text data in the 15-day dataset
text_data_15_days = train_data_15_days['msg_tx']
count_matrix_15_days = count_vectorizer.transform(text_data_15_days)
count_matrix_dense_15_days = count_matrix_15_days.toarray()

# Assign pseudo labels using each trained model
for name, model in trained_models.items():
    train_data_15_days[f'pseudo_label_{name.lower().replace(" ", "_")}'] = model.predict(count_matrix_dense_15_days)

# Now train_data_15_days contains pseudo labels assigned by each model in separate columns
# You can save train_data_15_days to a new Excel file or further process it as needed
