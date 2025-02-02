""" Beyonce Zhou
    ITP-449
    Final Project - NASA Kepler Exoplanet Candidate Classification
    Train and test an optimized Classification model (using cross validation), and analyze its results 
"""

#import libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import math

def main():
    # write your code here

    #load dataset
    file_path = 'cumulative_2023.11.07_13.44.30.csv'
    df_kepler = pd.read_csv(file_path,skiprows=41)

    #wrangling
    df_kepler = df_kepler.drop(columns=[col for col in df_kepler.columns if 'err' in col or col == 'koi_disposition'])
    df_kepler = df_kepler.dropna()

    #separate features and target
    X = df_kepler.drop(columns=['koi_pdisposition'])
    y = df_kepler['koi_pdisposition']

    #train test split 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=29)

    #define models and hyperparameter grids
    models = {
        'LogisticRegression': (LogisticRegression(),{}),
        'KNeighborsClassifier': (KNeighborsClassifier(),{'n_neighbors':list(range(1,int(1.5*np.sqrt(len(X_train)))+1))}),
        'DecisionTreeClassifier': (DecisionTreeClassifier(),{'criterion': ['entropy','gini'],'max_depth': list(range(3,16)),'min_samples_leaf': list(range(1,11))}),
        'SVC': (SVC(),{'kernel': ['rbf'],'C': [0.1,1,10,100],'gamma': [0.1,1,10]})
    }
    
    #helper function for cross validation and return best model
    def cross_validate_model(model,param_grid,X,y):
        pipeline=Pipeline([('scaler',StandardScaler()), ('model',model)])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions={'model__'+key: value for key, value in param_grid.items()},
            n_iter=max(1, math.ceil(0.1*math.prod([len(v) for v in param_grid.values()])))
        )
        search.fit(X,y)
        return search
    
    #perform cross validation on non pca dataset
    results = {}
    for name, (model,param_grid) in models.items():
        results[name] = cross_validate_model(model,param_grid,X_train,y_train)

    #select best non pca model
    best_non_pca_model = max(results,key=lambda name: results[name].best_score_)
    best_non_pca_score = results[best_non_pca_model].best_score_

    #perform PCA
    pca = PCA(n_components=.95)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train),index=X_train.index)
    X_test_pca = pd.DataFrame(pca.transform(X_test),index=X_test.index)

    #perform cross validation on pca dataset
    results_pca = {}
    for name, (model,param_grid) in models.items():
        results_pca[name] = cross_validate_model(model,param_grid,X_train_pca,y_train)

    #select best pca model
    best_pca_model = max(results,key=lambda name: results[name].best_score_)
    best_pca_score = results_pca[best_pca_model].best_score_

    #compare pca and nonpca
    if best_pca_score > best_non_pca_score:
        best_model = results_pca[best_pca_model]
        X_train_final, X_test_final = X_train_pca, X_test_pca
    else:
        best_model = results[best_non_pca_model]
        X_train_final,X_test_final = X_train,X_test
    
    #hyperparameter tuning with GridSearch
    best_model_param_grid = models[best_pca_model if best_pca_score > best_non_pca_score else best_non_pca_model][1]
    gridSearch = GridSearchCV(
        best_model.best_estimator_,
        param_grid = {'model__'+key: value for key, value in best_model_param_grid.items()},
    )
    gridSearch.fit(X_train_final,y_train)

    #evaluate final model
    y_pred = gridSearch.best_estimator_.predict(X_test_final)

    #confusion matrix
    conf_matrix = confusion_matrix(y_test,y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y_test))
    fig,axes = plt.subplots(1,1,figsize=(16,9))
    cm_disp.plot(ax=axes)
    fig.savefig('NASA Kepler Exoplanet Candidate Classification Confusion Matrix')

    #classification report
    print(classification_report(y_test,y_pred))

    #reflection questions
    print("1. Did PCA improve upon or not improve upon the results from when you did not use it? \n PCA did not improve the results from when I did not use it.")
    print ("Best Non PCA Score: ",best_non_pca_score)
    print ("Best PCA Score: ",best_pca_score)
    print("2. Why might this be? \n This might be because the PCA oversimplified the data and removed variance that could have been valuable for the model")
    print("3. Was your model able to classify objects equally across labels? How can you tell? \n Yes, it was able to classify objects equally across labels since the performance metrics are the same for both candidate and false positive.")
    print("4. Based on your results, which attribute most significantly influences whether or not an object is an exoplanet? (Hint: you have a way to quantitatively analyze this; it was covered in class.) \n koi_incl has a .34 correlation with koi_pdisposition, and is the most significant attribute in influencing whether an object is an exoplanet.")
    df_kepler_numeric = df_kepler
    df_kepler_numeric['koi_pdisposition'] = df_kepler_numeric['koi_pdisposition'].map({'CANDIDATE':1,'FALSE POSITIVE':0})
    correlation_matrix = df_kepler_numeric.corr()
    correlation_with_pdisposition = correlation_matrix['koi_pdisposition']
    print('Correlation matrix:' , correlation_with_pdisposition)
    print("5. Describe that attribute in your own words and why you believe it might be most influential. (This is an opinion question so the only way to get it wrong is to not actually reflect.) \n Since the inclination is positively correlated with koi_pdisposition being 'candidate', this means as incline angle increases, more likely that the object is an exoplanet candidate. This could occur because a higher inclination angle may indicate more interaction with the gravitational space of other bodies, such as the stars being monitored, making its detection as an exoplanet more likely.")

if __name__ == '__main__':
    main()
