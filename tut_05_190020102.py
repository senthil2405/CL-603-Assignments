"""=================================================== Assignment 5 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc. 

"""


""" Import the required libraries"""
# Start your code here
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
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
    x=x_input
    y = (x[0] - 1)**2 + (x[1] - 1)**2 - (x[0]*x[1])
    #y= x[0]**2 + x[1]**2 + (0.5*x[0]+x[1])**2 + (0.5*x[0]+x[1])**4
    
    # End your code here
    
    return y

def gradient(func, x_input):
  """
  --------------------------------------------------------------------------------------------------
  Write your logic for gradient computation in this function. Use the code from assignment 2.

  Input parameters:  
    func : function to be evaluated
    x_input: input column vector (numpy array of n dimension)

  Returns: 
    delF : gradient as a column vector (numpy array)
  --------------------------------------------------------------------------------------------------
  """
  # Start your code here
  # Use the code from assignment 2
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

def hessian(func, x_input):
  """
  --------------------------------------------------------------------------------------------------
  Write your logic for hessian computation in this function. Use the code from assignment 2.

  Input parameters:  
    func : function to be evaluated
    x_input: input column vector (numpy array)

  Returns: 
    del2F : hessian as a 2-D numpy array
  --------------------------------------------------------------------------------------------------
  """
  # Start your code here
  # Use the code from assignment 2
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

def backtracking(f, x ,p, rho = 0.8, alpha = 5, c = 0.1):
  grad=gradient(f,x)
  while f(x + alpha*p) > f(x) + c*alpha*(np.matmul(grad.T,p)):
    alpha = rho*alpha
  return alpha

def FRCG(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for FR-CG using in-exact line search. 

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
    max_iter=15000
    e=1e-6

    x_output = x_initial
    f_output = func(x_initial)
    num_iters = 0
    x_iter = [x_output]
    f_values = [f_output]
    grad = gradient(func,x_output)
    p = -grad
    for i in range(max_iter):
      alpha= backtracking(func,x_output,p)
      

      x_output = x_output + alpha*p
      x_iter.append(x_output)

      grad_output = gradient(func,x_output)
      t1 = np.matmul(grad_output.T,grad_output)
      t2 = np.matmul(grad.T,grad)
      beta = t1/t2
      p = -grad_output+beta*p

      num_iters += 1

      f_output = func(x_output)
      f_values.append(f_output)

      grad=gradient(func,x_output)
      grad_norm= np.linalg.norm(grad_output)**2
      if (grad_norm) < e:
        break
    if num_iters==max_iter:
      print('Maximum iterations reached but convergence did not happen')
    print('X versus iteration number for FRCG' )
    x0=[]
    x1=[]
    y1=[]
    for i in range(num_iters+1):
      x0.append(x_iter[i][0])
      x1.append(x_iter[i][1])

    #print(np.squeeze(x_iter).shape)
    plot_x_iterations(num_iters,[x0,x1])
    plot_func_iterations(num_iters,f_values)
    
    return [x_output, f_output, grad_output]

def plot_x_iterations(NM_iter, NM_x):
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

    ax1.plot(np.arange(0,NM_iter+1),NM_x[0],'go-')
    #ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('x1')

    ax2.plot(np.arange(0,NM_iter+1),NM_x[1],'b*-')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('x2')

    ax1.set_title('Variation of X versus Number of iterations of {}'.format('FRCG'))
    plt.show()
    # End your code here

def plot_func_iterations(NM_iter, NM_f):
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
    plt.title("function values vs No of Iterations for {}".format(FRCG))
    plt.show()
    


    # End your code here
    

"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array
    
"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output = FRCG(func, x_initial)
print("final value of x, and the corresponding f(x) and ∇f(x) are")
print(x_output)
print(f_output)
print(grad_output)