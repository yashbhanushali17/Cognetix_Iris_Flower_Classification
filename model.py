#cognetix internship project : 2 (foundation)

#IRIS CLASSIFICATION PROJECT

#step 1 : import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#step 2 :  load dataset


# iris.data does not contain column names, so we create names manually
columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
]

df=pd.read_csv('iris.data',names=columns)
print(df.head())
print(df.shape)
df.info()
print(df.describe())


#check missing values
print(df.isnull().sum())    

#numbers of species present
print(df['species'].value_counts())


#step 3 : EDA

#pairpoint  relationship view
sns.pairplot(df,hue='species')
plt.savefig('pairplot.png')
plt.show()

#heatmap 
numeric_values=df.drop('species',axis=1)

correlation=numeric_values.corr()

plt.figure(figsize=(8,6))
sns.heatmap(
    correlation,annot=True,cmap='coolwarm'
)
plt.title('correlation heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

#class distribution graph

plt.figure(figsize=(6,4))
df['species'].value_counts().plot(kind='bar')
plt.title('class distribution')
plt.xlabel('species')
plt.ylabel('count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

#step 4 : Encode target labels

encoder=LabelEncoder()
df['species']=encoder.fit_transform(df['species'])
print(df['species'].value_counts())

#step 5 : spliting of dataset

X=df.drop('species',axis=1)
y=df['species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

print(X_train.shape)
print(X_test.shape)

#feature scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#step 6 : model training

#logistic regression

models={
    "logistic_model":LogisticRegression(max_iter=200),

#decision tree model

"decision_tree_model":DecisionTreeClassifier(
    random_state=42
),

#Knn model 

"knn_model":KNeighborsClassifier(
    n_neighbors=5
)
}
results={}
for name,model in models.items():

    model.fit(X_train,y_train)
    
    y_pred=model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)

    results[name]={
        'model':model,
        'prediction':y_pred,
        'accuracy':accuracy,
        'confusion_matrix':cm
    }
    
    print(f'{name} accuracy: {round(accuracy,4)}')
    print(f'{name} confusion matrix: {cm}')

    print('classification report:')
    print(classification_report(y_test,y_pred,target_names=encoder.classes_))

    print('-----------------------------------')

#step 7 : visualization

#confusion matrix

for name ,result in results.items():

    cm=result['confusion_matrix']    

    plt.figure(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )
    plt.title(
        f'{name} confusion matrix'
    )
    plt.xlabel('actual species')
    plt.ylabel('predicted species')
    plt.tight_layout()
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.show()

#feature importance using decision tree


decision_tree = results["decision_tree_model"]["model"]

feature_importance = decision_tree.feature_importances_

plt.figure(figsize=(7,5))

plt.bar(X.columns,feature_importance)

plt.title('Decision Tree Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')

plt.tight_layout()
plt.savefig('decision_tree_feature_importance.png')
plt.show()


#step 8 : saving best model

best_model_name=max(results,key=lambda x:results[x]['accuracy'])

best_model=results[best_model_name]["model"]

joblib.dump(best_model,'iris_best_model.pkl')

print('best model : ',best_model_name)