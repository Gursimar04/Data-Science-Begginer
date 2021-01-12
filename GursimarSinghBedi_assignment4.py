import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('C:/Users/Predator/Desktop/iris.csv')
print(df.head())

print(df.isnull().sum())  # no null values

print(df.describe())  # no negative or absurd values

df = df.drop('Id', axis=1)  # dropping id column as it does not add any value to predictions
print(df.head())

print(df['Species'].value_counts())

species = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}  # converting to numeical data

numerical_df = df

numerical_df['Species'] = [species[val] for val in numerical_df['Species']]

print(numerical_df.head())

plt.figure(figsize=(20, 10))
sns.heatmap(numerical_df.corr(), annot=True)
plt.show()


# Create correlation matrix
corr_matrix = numerical_df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(to_drop)


numerical_df.drop(to_drop, axis=1)
x = numerical_df.drop('Species',axis=1)
y = numerical_df['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)
tree=DecisionTreeClassifier(random_state=27)

tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

clf_GS = GridSearchCV(tree, tree_para)

clf_GS.fit(x_train , y_train)
print(clf_GS.best_params_)
train_predict = clf_GS.predict(x_train)
test_predict = clf_GS.predict(x_test)

print("Train Accuracy:"+str(accuracy_score(y_train, train_predict)*100))
print("Test Accuracy:"+str(accuracy_score(y_test, test_predict)*100))

print("F1 Score on Train:"+str(f1_score(y_train, train_predict, average="macro")*100))
print("F1 Score on Test:"+str(f1_score(y_test, test_predict, average="macro")*100))


print("Confusion Matrix on Train:")
print(confusion_matrix(y_train, train_predict))

print("Confusion Matrix on Test:")
print(confusion_matrix(y_test, test_predict))
