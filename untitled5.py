# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MWtAKuzXvCvNGgoL9Ph98pQXHXi-L4QD
"""

# 1
import numpy as np
import pandas as pd
df = pd.read_excel("output.xlsx")
selected_columns = df.iloc[:, :-1]
selected_label = df.iloc[:, -1]
data=selected_columns.to_numpy()
label=selected_label.to_numpy()
print(label)

class1_data = data[selected_label == 0]
class2_data = data[selected_label == 1]
centroid1=class1_data.mean(axis=0)
centroid2=class2_data.mean(axis=0)
spread1=class1_data.std(axis=0)
spread2=class2_data.std(axis=0)
# print(class1_data)
meann=np.mean(data)
stdd=np.std(data)
print("mean of the data",meann)
print("centroids of the data",centroid1,centroid2)
print("std of data",stdd)
print("spread for the data",spread1,spread2)
print("euclidean dist between the 2 centroids",np.linalg.norm(centroid1 - centroid2))

# 2
import matplotlib.pyplot as plt
feature_column=df.iloc[:, 1]
feature_data=feature_column.to_numpy()
# print(feature_column)
meanf=np.mean(feature_data)
varf=np.var(feature_data)
plt.figure(figsize=(8,5))
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(meanf, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {meanf:.8f}")
plt.xlabel("Feature Values")
plt.ylabel("Frequency")
plt.title("Histogram of Selected Feature")
plt.legend()
plt.show()

# 3
def minkowski_dist(a,b,r):
  return np.sum(np.abs(a-b)**r)**(1/r)
fv1=df.iloc[1,:-1].to_numpy()
fv2=df.iloc[2,:-1].to_numpy()
dist=[]
r_values=[]
for i in range(1,10):
  dist.append(minkowski_dist(fv1,fv2,i))
  r_values.append(i)

plt.figure(figsize=(8, 5))
plt.plot(r_values, dist, marker='o', linestyle='-', color='b')
plt.xlabel("r Value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance Between Two Feature Vectors fro r=1 to 10")
plt.grid(True)
plt.show()

# 4
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

# 5
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data,label)

# 6
print(neigh.score(X_test, y_test))

# 7
print(neigh.predict(X_test))
print(y_test==neigh.predict(X_test))

# 8
neigh1=KNeighborsClassifier(n_neighbors=1)
neigh1.fit(data,label)
print(neigh1.score(X_test, y_test))
print(neigh.score(X_test, y_test))

# 9
from sklearn.metrics import confusion_matrix, classification_report
y_train_pred=neigh.predict(X_train)
y_test_pred=neigh.predict(X_test)

train_accuracy=neigh.score(X_train, y_train)
test_accuracy=neigh.score(X_test, y_test)

print("training accuracy:",train_accuracy)
print("testing accuracy:",test_accuracy)

train_conf_matrix=confusion_matrix(y_train,y_train_pred)
test_conf_matrix=confusion_matrix(y_test,y_test_pred)

print("Confusion Matrix for Train Data:\n", train_conf_matrix)
print("Confusion Matrix for Test Data:\n", test_conf_matrix)

precition_train=train_conf_matrix[0][0]/(train_conf_matrix[0][0]+train_conf_matrix[0][1])
precition_test=test_conf_matrix[0][0]/(test_conf_matrix[0][0]+test_conf_matrix[0][1])

print("precition for training data",precition_train)
print("precition for testing data",precition_test)

recall_train=train_conf_matrix[0][0]/(train_conf_matrix[0][0]+train_conf_matrix[1][0])
recall_test=test_conf_matrix[0][0]/(test_conf_matrix[0][0]+test_conf_matrix[1][0])

print("recall for training data",recall_train)
print("recall for testing data",recall_test)

f1_train=(2*train_conf_matrix[0][0])/(2*train_conf_matrix[0][0]+train_conf_matrix[0][1]+test_conf_matrix[1][0])
f1_test=(2*test_conf_matrix[0][0])/(2*test_conf_matrix[0][0]+test_conf_matrix[0][1]+test_conf_matrix[1][0])

print("f1 score for training data",f1_train)
print("f1 score for testing data",f1_test)


if train_accuracy > 0.98 and test_accuracy < 0.75:
    print("The model is Overfitting.")
elif train_accuracy < 0.70 and test_accuracy < 0.70:
    print("The model is Underfitting.")
else:
    print("The model is Generalized well (Regular fit).")