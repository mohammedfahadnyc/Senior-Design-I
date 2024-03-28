import pandas as pd
import nlpaug.augmenter.word as naw

# Assuming combined_df is your DataFrame with 'msg_tx' and 'outage_indicator' columns

# Initialize the augmentation method
augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

# Augment each class separately
augmented_data = []
for class_label in combined_df['outage_indicator'].unique():
    class_data = combined_df[combined_df['outage_indicator'] == class_label]['msg_tx']
    augmented_samples = augmenter.augment(class_data)
    augmented_data.extend([(text, class_label) for text in augmented_samples])

# Create a new DataFrame with augmented data
augmented_df = pd.DataFrame(augmented_data, columns=['msg_tx', 'outage_indicator'])

# Now augmented_df contains the augmented dataset
columns
# 
