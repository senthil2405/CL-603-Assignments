"""=================================================== Assignment 4 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc. 

"""

#%%
""" Import the required libraries"""
# Start your code here
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math

# End your code here


def func(x_input):
    """
    --------------------------------------------------------
    Write your logic to evaluate the function value. 

    Input parameters:
        x: input column vector (a numpy array of n dimension)

    Returns:
        y : Value of the function given in the problem at x.
        
    --------------------------------------------------------
    """
    
    # Start your code here
    x1,x2 = x_input[0],x_input[1]
    y = 2*np.exp(x1)*x2 + 3*x1*(x2**2)
    
    # End your code here
    
    return y

def gradient(x_input):
    h = 0.001
    grad_f = np.array([])
    for i in range(len(x_input)):
        e = np.array([np.zeros(len(x_input), dtype=int)]).T
        e[i][0] = 1
        del_f = (func(x_input + (h*e)) - func(x_input - (h*e)))/ (2*h)
        grad_f = np.append(grad_f, del_f)

    delF = np.array([grad_f]).T
    
        
    # End your code here

    return delF        

def hessian(x_input):
    n = len(x_input)
    del_x = np.full(shape=n, fill_value=0.001)
    del2F = np.array([]).reshape(0, n)
    for i in range(n):
        hess_f = np.array([])
        del_i = np.array([np.zeros(n)]).T
        del_i[i][0] = del_x[i]
        for j in range(n):
            del_j = np.array([np.zeros(n)]).T
            del_j[j][0] = del_x[j]
            if(i == j):
                a = x_input + del_i
                b = x_input - del_j
                value = (func(a) - (2*func(x_input)) + func(b))/(del_x[i]**2)
                hess_f = np.append(hess_f, value)
            else:
                a = x_input + del_i + del_j
                b = x_input - del_i - del_j
                c = x_input - del_i + del_j
                d = x_input + del_i - del_j
                value = (func(a) + func(b) - func(c) - func(d))/(4*del_x[i]*del_x[j])
                hess_f = np.append(hess_f, value)

        del2F = np.vstack([del2F ,hess_f])

    # End your code here
    
    return del2F

def plot_x_iterations(NM_iter, NM_x,title):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for plotting x_input versus iteration number i.e,
    x1 with iteration number and x2 with iteration number in same figure but as separate subplots. 

    Input parameters:  
        NM_iter : no. of iterations taken to converge (integer)
        NM_x: values of x at each iterations, a (num_interations X n) numpy array where, n is the dimension of x_input

    Output the plot.
    -----------------------------------------------------------------------------------------------------------------------------
    """
    # Start your code here
    fig,(ax1,ax2) = plt.subplots(2,1,figsize= (6,6),sharex = 'all')

    ax1.plot(np.arange(0,NM_iter+1),NM_x[0,:],'go-')
    #ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('x1')

    ax2.plot(np.arange(0,NM_iter+1),NM_x[1,:],'b*-')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('x2')

    ax1.set_title('Variation of X versus Number of iterations of {}'.format(title))

    # End your code here

def plot_func_iterations(NM_iter, NM_f,title):
    """
    ------------------------------------------------------------------------------------------------
    Write your logic to generate a plot which shows the value of f(x) versus iteration number.

    Input parameters:  
        NM_iter : no. of iterations taken to converge (integer)
        NM_f: function values at each iteration (numpy array of size (num_iterations x 1))

    Output the plot.
    -------------------------------------------------------------------------------------------------
    """
    # Start your code here
    fig = plt.figure(figsize = (6,6))

    plt.plot(np.arange(0,NM_iter+1),NM_f,'o--')
    
    plt.xlabel("No of Iterations")
    plt.ylabel("function values")
    plt.title("function values vs No of Iterations for {}".format(title))

    # End your code here

def backtracking(func,x):
    i = 0
    alpha = 5
    max_iter = 15000
    c = 0.1
    rho  = 0.8

    while(func(x-alpha*gradient(x)) > func(x) - c*alpha*np.dot(gradient(x).T,gradient(x)) and (i<max_iter)):
        alpha *= rho
        i += 1

    return alpha

def steepest_descent(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for steepest descent using in-exact line search. 

    Input parameters:  
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector(numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    
    # Start your code here
    x = x_initial
    
    tol = 1
    max_iter = 15000
    num_iterations = 0
    x_iter = x_initial
    func_values = func(x)

    while(norm(gradient(x))**2>10**-6 and num_iterations <= max_iter):
        gradient_step = backtracking(func,x)

        x = x - gradient(x)*gradient_step
        num_iterations += 1

        x_iter = np.hstack((x_iter,x))
        func_values = np.hstack((func_values,func(x)))
    

    x_output =x
    f_output = func(x)
    grad_output = gradient(x)
    # End your code here
    
    return x_output, f_output, grad_output
    
def backtracking_newton(func,x,max_iter):
    i = 0
    alpha = 5
    c = 0.1
    rho = 0.8


    while(func(x-alpha*(np.linalg.inv(hessian(x)).dot(gradient(x)))) > (func(x) - c*alpha*np.dot(gradient(x).T,(np.linalg.inv(hessian(x)).dot(gradient(x))))) and i<max_iter):
        alpha *= rho
        i += 1

    return alpha


def newton_method(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for newton method using in-exact line search. 

    Input parameters:  
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector (numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    # Start your code here
    x = x_initial
    grad = gradient(x)
    num_iter =0
    max_iter = 15000
    x_iter = x_initial
    func_values = func(x)

    while (num_iter <= max_iter and norm(gradient(x))**2 > 10**-6):
        if(np.linalg.det(hessian(x)) == 0):
            break
        alpha = backtracking_newton(func,x,max_iter)

        x -= alpha*(np.linalg.inv(hessian(x)).dot(gradient(x)))
        num_iter +=1
        x_iter = np.hstack((x_iter,x))
        func_values = np.hstack((func_values,func(x)))
    

    x_output = x
    f_output = func(x)
    grad_output = gradient(x)

    # End your code here
    
    return x_output, f_output, grad_output


def backtracking_bfgs(func,alpha, rho, c, x, max_iter,C):

    while (func(x - alpha*C.dot(gradient(x))) > func(x) - c*alpha*np.dot(np.transpose(gradient(x)),C.dot(gradient(x)))) and (i<max_iter):
        alpha = alpha*rho
        i = i+1
    
    return alpha

def quasi_newton_method(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for quasi-newton method with in-exact line search. 

    Input parameters:  
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector (numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    
    # Start your code here
    x = x_initial
    x_prev = x_initial

    i = 0
    H = np.eye(len(x))
    max_iter = 15000
    C = np.identity(2)
    alpha = 5
    rho = 0.8
    c = 0.1
    x_iter = x_initial
    func_values = func(x)

    while (i<max_iter and norm(gradient(x))**2 > 10**-6):
        alpha = backtracking_bfgs(func,alpha, rho, c, x_prev, max_iter,C)    
        x = x_prev - alpha*(C.dot(gradient(x_prev)))
        y = gradient(x) - gradient(x_prev)
        s = x - x_prev
        I = np.identity(2)

        C = (I - s.dot(y.T)/y.T.dot(s)).dot(C).dot(I - y.dot(s.T)/y.T.dot(s)) + s.dot(s.T)/y.T.dot(s)
        x_prev = x
        i = i+1 

        x_iter = np.hstack((x_iter,x))
        func_values = np.hstack((func_values,func(x)))
    

    x_output = x
    f_output = func(x)
    grad_output = gradient(x)

    # End your code here
    
    return x_output, f_output, grad_output



def iterative_methods(func, x_initial):
    """
     A function to call your steepest descent, newton method and quasi-newton method.
    """
    x_SD, f_SD, grad_SD = steepest_descent(func, x_initial)
    x_NM, f_NM, grad_NM = newton_method(func, x_initial)
    x_QN, f_QN, grad_QN = quasi_newton_method(func, x_initial)

    return x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN 
    
    
    
    
"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array
    
"""

# Define x_initial here

x_initial = np.array([1.5,1.5]).reshape((2,1))
x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN = iterative_methods(func, x_initial)
# %%
