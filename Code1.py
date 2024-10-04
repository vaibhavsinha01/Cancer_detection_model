from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
data = load_breast_cancer()

# Convert to DataFrame
data_table = pd.DataFrame(data.data, columns=data.feature_names)
data_table['target'] = data.target

print(data_table)

# Feature and target selection
X = data_table.drop(columns=['target'])  # Select all feature columns
y = data_table['target']  # Target variable

# Split the data into training and test sets
split = int(len(data_table) * 0.8)
x_train = X.iloc[:split]
y_train = y.iloc[:split]
x_test = X.iloc[split:]
y_test = y.iloc[split:]

# Apply scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # Only transform test data, not fit

# Initialize and train the model
reg = XGBClassifier()
reg.fit(x_train, y_train)

# Make predictions
y_pred = reg.predict(x_test)

# Calculate metrics
acc_score = accuracy_score(y_test, y_pred)
pre_score = precision_score(y_test, y_pred, average='binary')

# Print the scores
print(f"Accuracy Score: {acc_score}")
print(f"Precision Score: {pre_score}")

joblib.dump(reg,'cancer_detection.pkl')
loaded_model = joblib.load('cancer_detection.pkl')