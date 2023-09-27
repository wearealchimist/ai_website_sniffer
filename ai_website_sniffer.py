import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests
import matplotlib.pyplot as plt

url = input("Enter website :")
response = requests.get(url)
with open('dataset.csv', 'wb') as file:
    file.write(response.content)
    
data = pd.read_csv('dataset.csv')

print(data.head())
print(data.describe())

data = data.fillna(0)
data = pd.get_dummies(data, columns=['category'])

X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model precision : {accuracy * 100:.2f}%")

feature_importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(10), feature_importances[sorted_idx][:10], rotation=45)
plt.xticks(range(10), feature_names[sorted_idx][:10], rotation=45)
plt.xlabel('Important fuction')
plt.ylabel('Important')
plt.title('Important fuction of this model')
plt.show()
