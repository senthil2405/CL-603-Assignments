def func(x):
    #y=(x[0]-1)**2+(x[1]-1)**2-x[0]*x[1]
    #y=(x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2  
    # y =  x[0]+(2-x[1])**2
    y=(1-x[0])**2+100*(x[1]-x[0]**2)**2
    return y

def cost(x_values,x0,fraction=1):
    distance_vals = np.sqrt(np.sum((x_values - x0) ** 2, axis=1))
    sorted_distances = np.sort(distance_vals)
    n = len(sorted_distances)
    frac_n = int(n*fraction)
    frac_distances = sorted_distances[:frac_n]
    # Return mse of x_values from the
    return np.sum(frac_distances)

import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self,x_init):
        self.position  =  x_init
        self.velocity = np.zeros(len(x_init))

        self.p_best  = self.position.copy()
        self.pbest_val =  np.inf

    def  velocity_step(self,c1,c2,g_best,num_dimensions,w):
        r1 = np.random.rand(num_dimensions)
        r2 = np.random.rand(num_dimensions)

        self.velocity = w*self.velocity + c1*r1*(self.p_best-self.position) + c2*r2*(g_best-self.position)

    def position_step(self,search_space):
        self.position += self.velocity    
        self.position = np.clip(self.position,search_space[0],search_space[1])   

    def compute_best(self,func):
        self.f_val = func(self.position)

        if self.f_val< self.pbest_val:
            self.p_best = self.position
            self.pbest_val = self.f_val     


class PSO:
    def __init__(self,num_particles,num_dimensions,search_space,func):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.particles = [Particle(np.random.uniform(search_space[0],search_space[1],size = (self.num_dimensions,))) for _ in range(self.num_particles)]


        self.g_best = np.zeros(self.num_dimensions)
        self.gbest_val = float('inf')

        self.func = func
        self.search_space = search_space

    def train(self,max_iters,c1,c2):

        w_max = 0.9  # inertia weight
        w_min=0.4
        x_iterations = self.particles[0].position
        v_iterations = [self.particles[0].velocity]
        f_values = [func(self.particles[0].position)]
        tol_iterations =[10]

        x0 = [self.particles[0].position[0]]
        x1 = [self.particles[0].position[1]]

        for i in range(max_iters):

            for particle in self.particles:
                particle.compute_best(self.func)

                if particle.pbest_val < self.gbest_val:
                    self.g_best = particle.p_best
                    self.gbest_val  = particle.pbest_val

            for (i,particle) in enumerate(self.particles):
                w=w_max-(w_max-w_min)*i/max_iters
                particle.velocity_step(c1,c2,self.g_best,self.num_dimensions,w=0)
                particle.position_step(self.search_space)


            tol = cost([particle.position for particle in self.particles],self.g_best)

            x_iterations=np.vstack((x_iterations,self.particles[0].position))
            v_iterations=np.vstack((v_iterations,self.particles[0].velocity))       
            f_values=np.vstack((f_values,func(self.particles[0].position)))
            tol_iterations=np.vstack((tol_iterations,tol))

            x0.append(self.particles[0].position[0])
            x1.append(self.particles[0].position[1])

            
            if tol <= 10**-3:
                print('Tolerance reached breaking training')
                max_iters = i
                break 

        ##print(self.particles[0].position)
        print(x_iterations.shape)
        print(len(x0),len(x1),max_iters)
        #plot_x_iterations(max_iters+1,[x_iterations.T[:][0],x_iterations.T[:][1]])
        #plot_func_iterations(max_iters+1,f_values,"function")
        #plot_func_iterations(max_iters+1,tol_iterations,"tolerance",'log')
        return self.g_best, self.gbest_val,x_iterations
            


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

# Call the PSO function to optimize the objective function

pso = PSO(num_particles=30, num_dimensions=2,search_space=[-5.0,5.0],func = func)
gbest_pos, gbest_val,x_iterations = pso.train(max_iters = 1000,c1 = 2.0,c2 = 2.0)
# Print the results
print("Global best position:", gbest_pos)
print("Global best value:", gbest_val)

    




