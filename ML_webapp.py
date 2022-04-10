# importing Libraries

import sklearn
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# App headings
st.write(""" # Exploring Different ML Models and Dataset
Lets see which is best in them""")

# write dataset names and putting them in a sidebar(selecting box)
dataset_name = st.sidebar.selectbox( 'Select Dataset', 
('Iris', 'Breast Cancer', 'Wine'))

# write classifers names and putting them in a sidebar(selecting box)
classifier_name = st.sidebar.selectbox( 'Select Classifer', 
('SVM', 'KNN','Random Forest'))

# ab aik function define karna dataset ko load karne ke lia
def get_data(dataset_name):
    data= None
    if dataset_name == 'Iris':
        data= datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

# ab is function ko call karwayee ge or X, y ke equal rakh len ge
X,y = get_data(dataset_name)

# dataset ki shape ko app pe print karen ge
st.write('Dataset shape', X.shape)
st.write('Number of classes', len(np.unique(y)))


# ab ham classifer ke parameters ko user input ma add karen ge

def add_parameter_ui(classifier_name):
    params = dict()    # create empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C   # its the degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K   # it's no of nearest Neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth  # depth of every tree that grows in the forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # number of tree in the forest
    return params

# ab is function ko call karwae ge or params variable ke equal rakh len ge
params = add_parameter_ui(classifier_name)

# ab ham classifer bnaey gen base on classiferr_name and params
def get_classifier(classifier_name,params):
    clf= None
    if classifier_name =='SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth = params['max_depth'], random_state=1235)
    return clf

# ab id function ko call karwayee gen or clf ke equal rakh len ge
clf = get_classifier(classifier_name, params)

# ab dataset ko train test split lagate hen or 80/20 ratio rakh te hen
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)


# ab ham classifier ko train kar waye gen
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# model ki acuracy check karen ge or result ko app pa print karen ga
acc= accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"Accuracy =",acc)


# ## PLOT the dataset

# ab ham apne sare feature ko 2 dimensional plot pa draw kare using PCA 
pca= PCA(2)
X_projected = pca.fit_transform(X)

# ab ham apne data ko zero or 1 dimension may slice kare gen
x1= X_projected[ : , 0]
x2= X_projected[ : , 1]

fig= plt.figure()
plt.scatter(x1, x2, c=y, alpha= 1, cmap= 'viridis')

plt.xlabel('Princpal Component 1')
plt.ylabel('Princpal Component 1')
plt.colorbar()

# plot ko show karwayee gee
st.pyplot(fig)