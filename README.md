# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.
## Program:
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by :chadalawada jaswanth

Reg No : 212221040030
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
1.Array value of x :

![image](https://user-images.githubusercontent.com/94836154/233408486-c5f0f4f0-2d14-4f07-9371-59ca4dd143f5.png)

2.Array value of y :

![image](https://user-images.githubusercontent.com/94836154/233408671-0c1049df-b1ff-4841-b9b7-24e4b2807e36.png)

3.Exam 1 & 2 score graph :

![image](https://user-images.githubusercontent.com/94836154/233408756-f337a328-7cf5-4339-a1d1-e88a6c52e884.png)

4.Sigmoid graph :

![image](https://user-images.githubusercontent.com/94836154/233408940-255740e9-b302-4491-97c1-1a1b01048b44.png)

5.J and grad value with array[0,0,0] :

![image](https://user-images.githubusercontent.com/94836154/233409092-4cd93df2-0fd5-4445-a782-b1cb84de22f8.png)

6.J and grad value with array[-24,0.2,0.2] :

![image](https://user-images.githubusercontent.com/94836154/233409233-809dd6e3-f9dc-4957-bede-7b2fa7f59afa.png)

7.res.function & res.x value :

![image](https://user-images.githubusercontent.com/94836154/233409354-6ded5ad8-c4d9-4de4-94b0-cd3b3cd61e35.png)

8.Decision Boundary graph :

![image](https://user-images.githubusercontent.com/94836154/233409504-4d0a4c34-e01b-45bd-bc80-12aff74f437b.png)

9.probability value :

![image](https://user-images.githubusercontent.com/94836154/233409652-9d844061-2942-45e9-9e65-c55188e83eb2.png)

10.Mean prediction value :

![image](https://user-images.githubusercontent.com/94836154/233409791-33791e1f-a17d-442f-b128-b1aa1800358c.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
