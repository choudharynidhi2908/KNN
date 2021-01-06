import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier as KNC
#from sklearn.neighbors import KNeighborsClassifier as KNC


glass = pd.read_csv('C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\KNN\\glass.csv')

train,test =train_test_split(glass,test_size = 0.2)


train_glass_x = train.iloc[:,0:9]

train_glass_y = train.iloc[:,9]


test_glass_x = test.iloc[:,0:9]
test_glass_y = test.iloc[:,9]


acc = []

print(len(train_glass_x))
print(len(train_glass_y))
print(len(test_glass_x))
print(len(test_glass_y))

print('Model Started............')
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9]) 
    train_acc=np.mean(neigh.predict(train_glass_x)==train_glass_y)
    test_acc=np.mean(neigh.predict(test_glass_x)==test_glass_y)
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt 

plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.legend(["train_acc","test_acc"])
plt.show()









