"""=================================================== Assignment 6 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc. 

"""


""" Import the required libraries"""
# Start your code here
import numpy as np
import matplotlib.pyplot as plt

# End your code here


def func(x_input):
    """
    --------------------------------------------------------
    Write your logic to evaluate the function value. 

    Input parameters:
        x: input column vector (a numpy array of n x 1 dimension)

    Returns:
        y : Value of the function given in the problem at x.
        
    --------------------------------------------------------
    """
    
    # Start your code here
    #y = (x_input[0]-1)**2 + (x_input[1]-1)**2 - x_input[0]*x_input[1]
    y = abs(x_input[0])**2 + abs(x_input[1])**3
    
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
        
def TRPD(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for Trust Region - Powell Dogleg Method.

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
    eta = 0.2
    del0 = 0.5
    delta = 1

    x = x_initial
    max_iters = 15000
    tol = 1e-6
    iters = 0

    x_iter = [x]
    f_values = [func(x)]

    while iters < max_iters:
      g = gradient(func,x)
      H = hessian(func,x)

      pk = -(np.dot(g.T,g)/np.dot(g.T,np.dot(H,g)))*g
      pn = -np.dot(np.linalg.inv(H).T,g)

      if np.linalg.norm(pn)**2 <= delta:
        p_optimal = pn
      elif np.linalg.norm(pk) >= delta:
        p_optimal = (delta/np.linalg.norm(pk))*pk
      else:
        a = np.dot((pk-pn).T,pk-pn)
        b = 2*np.dot((pn-pk).T,pk)
        c = np.dot(pk.T,pk) - delta**2
        t = (-b+np.sqrt(b**2-4*a*c))/(2*a)

        p_optimal = pk + t*(pn-pk)

      m_optimal = func(x) + np.dot(g.T,p_optimal) + 0.5*np.dot(p_optimal.T,np.dot(H,p_optimal))
      rho = (func(x) - func(x+p_optimal))/(func(x)-m_optimal)

      if rho < 0.25:
        delta = 0.25*delta
      elif rho > 0.75 and abs(np.linalg.norm(p_optimal)-delta) < tol:
        delta = 2*delta

      if rho > eta:
        x = x + p_optimal
    
      iters += 1

      x_iter.append(x)
      f_values.append(func(x))

      if np.linalg.norm(gradient(func,x))**2 < tol:
        break
    
    if iters ==  max_iters:
      print('Max no of iterations reached and didnot converge')
    
    x0=[]
    x1=[]
    y1=[]
    for i in range(iters+1):
      x0.append(x_iter[i][0])
      x1.append(x_iter[i][1])

    #print(np.squeeze(x_iter).shape)
    plot_x_iterations(iters,[x0,x1])
    plot_func_iterations(iters,f_values)
    
    # End your code here

    return [x, func(x), gradient(func,x)]

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

    ax1.set_title('Variation of X versus Number of iterations of {}'.format('Dogleg Method'))
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
    plt.title("function values vs No of Iterations for {}".format('Dogleg Method'))
    plt.show()

    # End your code here



    
"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array
    
"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output = TRPD(func, x_initial)

print("\n\nTrust Region - Powell Dogleg Method:")
print("-"*40)
print("\nFunction converged at x = \n",x_output)
print("\nFunction value at converged point = \n",f_output)
print("\nGradient value at converged point = \n",grad_output)