import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import time
start_time = time.time()

def StochasticGD(x,y,alpha,iterations):
    theta0 = 0
    theta1 = 0
    c = 0
    while c < iterations:
        for i in range(0,len(x)):
            hx = theta0 + theta1 * x[i]
            theta0 = theta0 - alpha *2*(hx-y[i])
            theta1 = theta1 - alpha *2*(hx-y[i]) * x[i]
        c+=1
    return theta0, theta1

x_points = np.array([1,1,2,3,4,5,6,7,8,9,10,11])
y_points = np.array([1,2,3,1,4,5,6,4,7,10,15,9])

plt.scatter(x_points, y_points)
t0,t1 = StochasticGD(x_points,y_points,0.0021,2)
print("Theta0: ",t0)
print("Theta1: ",t1)

xline = np.array(range(0, 15))
yline = t0 + t1 * xline
plt.plot(xline, yline)

yhad = t0 + t1 * x_points
r2 = r2_score(y_points, yhad)
print("R2 Score : ",r2)

plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
