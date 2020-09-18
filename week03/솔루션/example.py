import numpy as np
# 1. Steepest Descent
def steepest_descent_2d(func, gradx, grady, x0, MaxIter=10, learning_rate=0.25):
    for i in range(MaxIter):
        grad = np.array([gradx(*x0), grady(*x0)])
        x1 = x0 - learning_rate * grad
        x0 = x1
    return x0

f = lambda x,y : 3 * (x - 2)**2 + (y - 2)**2
grad_x = lambda x,y : 6 * (x - 2)
grad_y = lambda x,y : 2 * (y - 2)

x0 = np.array([-2.0, -2.0])
learning_rate = 0.1
MaxIter = 100
xopt = steepest_descent_2d(f, grad_x, grad_y, x0,
                MaxIter=MaxIter, learning_rate=learning_rate)

print(xopt)

# 2. Newton method
def newton_descent_2d(func, gradx, grady, hessian, x0, MaxIter=10, learning_rate=1):
    for i in range(MaxIter):
        grad = np.array([gradx(*x0), grady(*x0)])
        hess = hessian(*x0)
        delx = np.linalg.solve(hess, grad)
        x1 = x0 - learning_rate * delx
        x0 = x1
    return x0

f = lambda x,y : 3 * (x - 2)**2 + (y - 2)**2
grad_x = lambda x,y : 6 * (x - 2)
grad_y = lambda x,y : 2 * (y - 2)
hessian = lambda x,y : np.array([[6., 0.],[0, 2.]])
x0 = np.array([-2.0, -2.0])
xopt = newton_descent_2d(f, grad_x, grad_y, hessian, x0)

print(xopt)

# 3. BFGS method
def bfgs_method_2d(func, gradx, grady, x0, MaxIter=10, learning_rate=1):
    B0 = np.eye(len(x0))
    for i in range(MaxIter):
        grad = np.array([gradx(*x0), grady(*x0)])
        p0 = -np.linalg.solve(B0, grad)
        delx = learning_rate * p0
        x1 = x0 + delx
        y0 = (np.array([gradx(*x1), grady(*x1)]) - grad).reshape(-1,1)
        B1 = B0 + np.dot(y0, y0.T) / np.dot(y0.T, delx) \
                - np.dot(np.dot(B0, delx).reshape(-1,1), np.dot(delx, B0).reshape(-1,1).T) \
                / np.dot(np.dot(B0, delx), delx)
        x0 = x1
        B0 = B1
    return x0

# Define functions for the problem
f = lambda x,y : 3 * (x - 2)**2 + (y - 2)**2
grad_x = lambda x,y : 6 * (x - 2)
grad_y = lambda x,y : 2 * (y - 2)
hessian = lambda x,y : np.array([[6., 0.],[0., 2.]])

# Tune parameters(Use default values for MaxIter, learning_rate)
x0 = np.array([-2.0, -2.0])
xopt = bfgs_method_2d(f, grad_x, grad_y, x0, MaxIter=6)

# Result will be [ 2.  2.]
print(xopt)