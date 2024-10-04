from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix,recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

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
model = Sequential()
model.add(Dense(128,input_dim=x_train.shape[1],activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))
# model.add(Dense(1,activation='sigmoid'))

# Make predictions
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_test,y_test))

# Calculate metrics
loss,accuracy = model.evaluate(x_test,y_test)
print(f'Test Accuracy: {accuracy:.4f}')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis=1)

acc_score = accuracy_score(y_test, y_pred_classes)
pre_score = precision_score(y_test, y_pred_classes, average='binary')
rec_score = recall_score(y_test,y_pred_classes)
matrix = confusion_matrix(y_test,y_pred_classes)
# Print the scores
print('The accuracy and precision score are:')
print(f"Accuracy Score: {acc_score:2f}")
print(f"Precision Score: {pre_score:2f}")
print(f"Recall score :{rec_score:2f}")
print(matrix)

model.save('cancer_detection_model.h5')