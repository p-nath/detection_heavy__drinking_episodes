from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def modelTrain(X, target, seed, n):
  # Create a based model
  clf = RandomForestClassifier(n_estimators=100, max_depth=n)
  # Instantiate the grid search model
  scaler = MinMaxScaler()
  X = scaler.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=seed)
  clf.fit(X_train, y_train)	
  y_pred = clf.predict(X_test)
  return metrics.accuracy_score(y_test, y_pred)*100

final_data = pd.read_csv('final_data.csv')
target = final_data['label']
final_data = final_data.drop('label', axis=1)
final_data = final_data.drop('Unnamed: 0', axis=1)
n=1
accuracies = []

for i in range(20):
  accuracies.append(modelTrain(final_data, target, 41, n))
  n += 1
  print("Iteration",i,", max_depth=", n)

depths = [i for i in range(1, 21)]
ax = sns.lineplot(x=depths, y=accuracies)
ax.set(title="max depth v/s accuracy")
ax.set(xlabel='Max Depth of Random Forest', ylabel='Accuracy over Testing Set (%)')
plt.show()

# n = 20 got the highest accuracy
acc_20 = []
for i in range(10):
  acc_20.append(modelTrain(final_data, target, np.random.randint(0,100), 20))