#Importing Necessary Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 

#Loading Dataset
iris = load_iris()

#Creating dataframe of the loaded dataset and Displaying first few rows
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

df.head()

#Splitting the dataset into features(X) and target(y)
X = df.drop('species', axis=1)
y = df['species']

#Splitting dataset into Training and Testing sets (70:30)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

#Initializing Classifier (DTC)
classifier = DecisionTreeClassifier()

#Training Decision Tree Classifier
classifier.fit(X_train,y_train)

#Making Predictions based on the testing data
y_pred = classifier.predict(X_test)

#Evaliating the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Classification Report: {class_report}")

#Confusion Matrix
labels = ['Setosa','Versicolor','Virginica']
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels= iris.target_names)

# Add labels and title
plt.xlabel('-- Predicted --')
plt.ylabel('-- Actual --')
plt.title('Confusion Matrix')
plt.show()

#Visualizing the Decision Tree
plt.Figure(figsize=(12,8))
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Flower Classification")
plt.show()