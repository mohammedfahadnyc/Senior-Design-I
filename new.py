from imblearn.over_sampling import ADASYN
from sklearn.utils import shuffle

# Assuming your combined DataFrame is named 'combined_df'
# Separate features (X) and target variable (y)
X = combined_df.drop(columns=['outage_indicator'])
y = combined_df['outage_indicator']

# Apply ADASYN to generate synthetic samples
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Combine original and synthetic samples
balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='outage_indicator')], axis=1)

# Optionally, shuffle the dataset
balanced_df = shuffle(balanced_df)

# Now 'balanced_df' contains a balanced dataset with ADASYN applied
