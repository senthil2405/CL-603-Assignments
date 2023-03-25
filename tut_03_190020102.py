"""============================================ Assignment 3: Newton Method ============================================"""

""" Import the required libraries"""
# Start you code here
import numpy as np
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

def newton_method(func, x_initial):
  """
  -----------------------------------------------------------------------------------------------------------------------------
  Write your logic for newton method in this function. 

  Input parameters:  
    func : input function to be evaluated
    x_initial: initial value of x, a column vector (numpy array)

  Returns:
    x_output : converged x value, a column vector (numpy array)
    f_output : value of f at x_output
    grad_output : value of gradient at x_output
    num_iterations : no. of iterations taken to converge (integer)
    x_iterations : values of x at each iterations, a (num_interations x n) numpy array where, n is the dimension of x_input
    f_values : function values at each iteration (numpy array of size (num_iterations x 1))
  -----------------------------------------------------------------------------------------------------------------------------
  """
  # Write code here
  x = x_initial
  x_iterations = x
  fval = func(x_initial)
  f_values = fval

  tol = 1
  max_iter = 10000
  num_iterations =0
  #print(hessian(func, x).shape)
  while (tol**2 > 10**-6 and num_iterations <= max_iter):
    gradient_f  = gradient(func,x)
    hessian_f = hessian(func,x)
    #print(hessian_f.shape)
    hessian_invf = np.linalg.inv(hessian_f)

    x -= np.matmul(hessian_invf,gradient_f)
    f = func(x)

    gradient_f  = gradient(func,x)
    tol = np.sqrt(np.dot(gradient_f.T,gradient_f))

    x_iterations = np.hstack((x_iterations,x))
    f_values = np.hstack((f_values,f))
    num_iterations += 1

  if num_iterations == max_iter:
     print('Max iterations reached and convergence didnot happen')
     
  x_output  = x
  grad_output = gradient(func,x_output)
  f_output = func(x_output)

  # End your code here

  return x_output, f_output, grad_output, num_iterations, x_iterations, f_values


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

  ax1.plot(np.arange(0,NM_iter+1),NM_x[0,:],'go-')
  #ax1.set_xlabel('Number of iterations')
  ax1.set_ylabel('x1')

  ax2.plot(np.arange(0,NM_iter+1),NM_x[1,:],'b*-')
  ax2.set_xlabel('Number of iterations')
  ax2.set_ylabel('x2')

  ax1.set_title('Variation of X versus Number of iterations of Newtons Method')

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
  plt.title("function values vs No of Iterations")

  # End your code here



"""--------------- Main code: Below code is used to test the correctness of your code ---------------"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output, num_iterations, x_iterations, f_values = newton_method(func, x_initial)

print("\nFunction converged at x = \n",x_output)
print("\nFunction value at converged point = \n",f_output)
print("\nGradient value at converged point = \n",grad_output)

plot_x_iterations(num_iterations, x_iterations)

plot_func_iterations(num_iterations, f_values)
# %%
