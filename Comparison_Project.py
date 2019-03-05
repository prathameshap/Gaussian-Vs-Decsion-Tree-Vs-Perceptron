# improt dependencies
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# read data
df_main = pd.read_csv('Dataset.csv')

df_main.head()

df_main.describe()

df_main.info()

# calculate the correlation matrix
corr = df_main.corr()
print(corr.columns)

# plot the correlation heatmap
sns.heatmap(corr)

df_main.columns

for i in df_main['Salary']:
    print("-",i,"-")
    print(str.strip(i))
    print(i.strip() is "<=50K")
    print(i.strip() == "<=50K")
    print((i == "<=50K"))
    break
    
 df_main['Salary'] = [0 if salary.strip() =="<=50K" else 1 if salary.strip()==">50K" else salary for salary in df_main['Salary']]
 
 # data cleaning
for column in df_main.columns:
    df_main[column] = [value.strip() if type(value) == str else value for value in df_main[column]]
    
df_main.head(10)

wc_enc = preprocessing.LabelEncoder()
X = df_main['WC']
wc_enc.fit(X.values)
df_main['WC'] = wc_enc.transform(df_main['WC'].values)

el_enc = preprocessing.LabelEncoder()
X = df_main['EL']
el_enc.fit(X.values)
df_main['EL'] = el_enc.transform(df_main['EL'].values)

ms_enc = preprocessing.LabelEncoder()
X = df_main['MS']
ms_enc.fit(X.values)
df_main['MS'] = ms_enc.transform(df_main['MS'].values)

occ_enc = preprocessing.LabelEncoder()
X = df_main['Occ']
occ_enc.fit(X.values)
df_main['Occ'] = occ_enc.transform(df_main['Occ'].values)

rs_enc = preprocessing.LabelEncoder()
X = df_main['RS']
rs_enc.fit(X.values)
df_main['RS'] = rs_enc.transform(df_main['RS'].values)

gender_enc = preprocessing.LabelEncoder()
X = df_main['Gender']
gender_enc.fit(X.values)
df_main['Gender'] = gender_enc.transform(df_main['Gender'].values)

df_main.head()

columns_to_use = ['Age', 'WC', 'EL', 'Year', 'MS', 'Occ', 'RS', 'Gender', 'CG', 'CL', 'Hours']

X = df_main[columns_to_use]
y = df_main['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

def train_with_folds(clf):
    # split array in k(number of folds) sub arrays
    X_folds = np.array_split(X_train, 3)
    y_folds = np.array_split(y_train, 3)
    
    scores = list()
    models = list()
    for k in range(3):

        # We use 'list' to copy, in order to 'pop' later on
        X_train_fold = list(X_folds)
        # pop out kth sub array for testing
        X_test_fold  = X_train_fold.pop(k)
        # concatenate remaining sub arrays for training
        X_train_fold = np.concatenate(X_train_fold)

        # same process for y
        y_train_fold = list(y_folds)
        y_test_fold  = y_train_fold.pop(k)
        y_train_fold = np.concatenate(y_train_fold)

        clf = clf.fit(X_train_fold, y_train_fold)
        scores.append(clf.score(X_test_fold, y_test_fold))
        models.append(clf)

    print(scores)
    
%Gaussain Naive Bayes 
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
print("mean accuracy ",gnb.score(X_test, y_test))

train_with_folds(clf=GaussianNB())

%Decision Tree Classifier

clf = DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X_train, y_train)
print("mean accuracy ", clf.score(X_test, y_test))

clf = DecisionTreeClassifier(max_depth=10)
train_with_folds(clf)

%Multilayer Preceptrons 

clf = MLPClassifier(solver='adam', activation='tanh', alpha=1e-5, hidden_layer_sizes=(15, 5), random_state=43)
clf = clf.fit(X_train, y_train)
print("mean accuracy ", clf.score(X_test, y_test))

clf = MLPClassifier(solver='adam', activation='tanh', alpha=1e-5, hidden_layer_sizes=(9, 2), random_state=43)
train_with_folds(clf)