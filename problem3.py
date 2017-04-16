import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
csvfile = file(sys.argv[2],'wb')
data = pd.read_csv(sys.argv[1])
for index, row in data.iterrows():
    if row['label'] == 0:
        plt.plot(row['A'],row['B'],'bo')
    if row['label'] == 1:
        plt.plot(row['A'],row['B'],'ro')
#plt.show()
x_df = pd.DataFrame(data.A)
x_df['B'] = pd.DataFrame(data.B)
y_df = pd.DataFrame(data.label)
X = np.array(x_df)
y = np.array(y_df).flatten()
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,stratify = y)
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
#print X_train
#print X_test

def lin_svc():
    svm = SVC(kernel='linear')
    search_space = [{'C':np.array([0.1,0.5,1,5,10,50,100])}]
    gridsearch = GridSearchCV(svm,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['svm_linear',cv_performance,test_performance])

def poly_svc():
    svm = SVC(kernel='poly')
    search_space = [{'C':np.array([0.1,1,3]),'degree':np.array([4,5,6]),'gamma':np.array([0.1,1])}]
    gridsearch = GridSearchCV(svm,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['svm_polynomial',cv_performance,test_performance])
    
def rbf_svc():
    svm = SVC(kernel='rbf')
    search_space = [{'C':np.array([0.1,0.5,1,5,10,50,100]),'gamma':np.array([0.1,0.5,1,3,6,10])}]
    gridsearch = GridSearchCV(svm,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['svm_rbf',cv_performance,test_performance])
    
def log_reg():
    logreg = LogisticRegression()
    search_space = [{'C':np.array([0.1,0.5,1,5,10,50,100])}]
    gridsearch = GridSearchCV(logreg,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['logistic',cv_performance,test_performance])
    
def k_neigh():
    neigh = KNeighborsClassifier()
    search_space = [{'n_neighbors':np.array(range(1,51)),'leaf_size':np.array(range(5,65,5))}]
    gridsearch = GridSearchCV(neigh,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['knn',cv_performance,test_performance])
    
def d_t():
    dt = DecisionTreeClassifier()
    search_space = [{'max_depth':np.array(range(1,51)),'min_samples_split':np.array(range(2,11))}]
    gridsearch = GridSearchCV(dt,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['decision_tree',cv_performance,test_performance])
    
def rand_forest():
    ranfor = RandomForestClassifier()
    search_space = [{'max_depth':np.array(range(1,51)),'min_samples_split':np.array(range(2,11))}]
    gridsearch = GridSearchCV(ranfor,param_grid = search_space,refit = True, cv = 5) 
    gridsearch.fit(X_train,y_train)
    print ('Best parameter:%s'%str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(X_test,y_test)
    print ('Cross-validation accuracy score: %0.3f,'' test accuracy score: %0.3f'%(cv_performance,test_performance))
    csv.writer(csvfile).writerow(['random_forest',cv_performance,test_performance])  

def main():
    print 'SVM with Linear Kernel:'
    lin_svc()
    print 'SVM with Polynomial Kernel:'
    poly_svc()
    print 'SVM with RBF Kernel:'
    rbf_svc()
    print 'Logistic Regression:'
    log_reg()
    print 'k-Nearest Neighbors:'
    k_neigh()
    print 'Decision Tree:'
    d_t()
    print 'Random_Forest:'
    rand_forest()
    
if __name__ =="__main__":
	main()
		