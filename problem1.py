import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
    
csvfile = file(sys.argv[1],'rb')
input = csv.reader(csvfile)
data = []
for row in input:
    row = map(int,row)
    data.append(row)
#print data


csvfile2 = file(sys.argv[2],'wb')
b = 0
w1 = 0
w2 = 0

def f(b,w1,w2,i):
    if (b + w1*data[i][0] + w2*data[i][1]) > 0:
        return 1
    elif (b + w1*data[i][0] + w2*data[i][1]) <= 0:
        return -1
    
flag = 1  
while flag != 0:
    flag = 0
    for i in range(len(data)):
        if data[i][2]*f(b,w1,w2,i) <= 0:
            w1 = w1 + data[i][2]*data[i][0]
            w2 = w2 + data[i][2]*data[i][1]
            b = b + data[i][2]*1       
            
    for i in range(len(data)):
        if data[i][2]*f(b,w1,w2,i) <= 0:
            flag = flag +1
    
    print w1, w2, b
    csv.writer(csvfile2).writerow([w1,w2,b])

print w1,w2,b
csv.writer(csvfile2).writerow([w1,w2,b])
for row in data:
    if row[2] == 1:
        plt.plot(row[0],row[1],'bo')
    if row[2] == -1:
        plt.plot(row[0],row[1],'ro')

#plt.plot([0,-b/w1],[-b/w2,0], 'g')
#plt.show()
sys.exit() 

        