import pandas as pd
import nlpaug.augmenter.word as naw

# Assuming combined_df is your DataFrame with 'msg_tx' and 'outage_indicator' columns
# Assuming you want to oversample class 1 and class 2 to balance the dataset

# Separate the data into different classes
class_0 = combined_df[combined_df['outage_indicator'] == 0]
class_1 = combined_df[combined_df['outage_indicator'] == 1]
class_2 = combined_df[combined_df['outage_indicator'] == 2]

# Determine the target number of samples for each class (assuming you want to balance with class 0)
target_samples = len(class_0)

# Initialize the augmentation method
augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

# Oversample class 1 and class 2
oversampled_class_1 = class_1
while len(oversampled_class_1) < target_samples:
    augmented_samples = augmenter.augment(class_1['msg_tx'])
    oversampled_class_1 = pd.concat([oversampled_class_1, pd.DataFrame({'msg_tx': augmented_samples, 'outage_indicator': 1})], ignore_index=True)

oversampled_class_2 = class_2
while len(oversampled_class_2) < target_samples:
    augmented_samples = augmenter.augment(class_2['msg_tx'])
    oversampled_class_2 = pd.concat([oversampled_class_2, pd.DataFrame({'msg_tx': augmented_samples, 'outage_indicator': 2})], ignore_index=True)

# Concatenate all classes together
balanced_df = pd.concat([class_0, oversampled_class_1, oversampled_class_2], ignore_index=True)

# Now balanced_df contains the balanced dataset with oversampled classes
