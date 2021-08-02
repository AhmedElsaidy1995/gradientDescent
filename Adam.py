import numpy as np
import matplotlib.pyplot as plt

X_input = np.linspace(0, 20, 50)
y_output = -1 * X_input + 2


def adam(x,y,alpha,beta1,beta2,iterations):
    theta = np.zeros(2)
    v = np.zeros(2)
    m = np.zeros(2)
    epsilon = 10**-8
    c = 0
    x = np.c_[x, np.ones((len(y),1))]

    costArray = []
    thetaArray = []
    hypothesis = []
    gredientNorm = 1
    costChange = 1

    while c < iterations and gredientNorm > 10**-5 and costChange > 10**-5 :
        #Calculate Hypothesis
        hx = x.dot(theta)
        hypothesis.append(hx)
        #Calculate Cost Function
        error = hx-y
        costFunction = np.sum((error**2))/2*len(y)
        costArray.append(costFunction)
        if c > 1 :
            costChange = abs(costArray[c-1] - costArray[c-2])
        #Calculate Gredient
        gredient = (x.transpose() @ error)/ len(y)
        gredientNorm = np.linalg.norm(gredient)
        #Update Thetas
        m = beta1 *m + (1-beta1)*gredient
        v = beta2 *v + (1-beta2)*gredient**2
        if c > 0 :
            m = m / (1-beta1**c)
            v = v / (1-beta2**c)
        theta = theta - ((alpha /(np.sqrt(v) + epsilon)) * m)
        thetaArray.append(theta)

        c+=1
    return np.array(thetaArray) , costArray , hypothesis , theta , c

all_thetas,cost,all_hypo,theta , iter= adam(x=X_input,y=y_output,alpha=0.01,beta1=0.9,beta2=0.9,iterations=30000)
print(iter)

from sklearn.metrics import r2_score
X_points = np.c_[X_input, np.ones((len(X_input),1))]
yhad = X_points.dot(theta)
r2 = r2_score(y_output, yhad)
print("R2 Score : ",r2)

iterations = np.array(range(1, iter+1))
plt.plot(iterations, cost)
plt.show()