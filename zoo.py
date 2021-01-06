import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier as KNC 
from sklearn.model_selection import train_test_split

zoo = pd.read_csv('C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\KNN\\Zoo.csv')
train,test = train_test_split(zoo,test_size= 0.3)

train_x = zoo.iloc[:,1:17]
train_y = zoo.iloc[:,17]
test_x = zoo.iloc[:,1:17]
test_y = zoo.iloc[:,17]
acc =[]
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train_x,train_y)
    train_acc = np.mean(neigh.predict(train_x)==train_y)
    test_acc = np.mean(neigh.predict(test_x)==test_y)
    acc.append([train_acc,test_acc])

print(acc)

plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
plt.legend(['train','test'])
plt.show()

