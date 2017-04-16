import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv(sys.argv[1],names = ['age','weight','height'])
#print data
csvfile = file(sys.argv[2],'wb')


#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#data.age = (data.age-data.age.mean())/np.std(data.age,0)
#data.weight = (data.weight-data.weight.mean())/np.std(data.weight,0)
data.age = (data.age-data.age.mean())/data.age.std()
data.weight = (data.weight-data.weight.mean())/data.weight.std()
x_df = pd.DataFrame(data.age)
x_df['weight'] = pd.DataFrame(data.weight)
x_df.insert(0,'intercept',1)
y_df = pd.DataFrame(data.height)
#print x_df
#ax.scatter(x_df['age'],x_df['weight'],y_df)
#plt.show()

iterations = 100
alphas = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
iteration = 1000
alpha_new = 0.8

x = np.array(x_df)
y = np.array(y_df).flatten()
#print x
#print y

for alpha in alphas:
    beta = np.array([0,0,0])
    for i in range(iterations):
        beta = beta - alpha*x.T.dot(x.dot(beta)-y)/len(y)
        #print beta
    print alpha,iterations,beta[0],beta[1],beta[2]
    csv.writer(csvfile).writerow([alpha,iterations,beta[0],beta[1],beta[2]])
    
beta = np.array([0,0,0])    
for i in range(iteration):
    beta = beta - alpha_new*x.T.dot(x.dot(beta)-y)/len(y)
print alpha_new,iteration,beta[0],beta[1],beta[2]
csv.writer(csvfile).writerow([alpha,iterations,beta[0],beta[1],beta[2]])
#print beta
sys.exit()