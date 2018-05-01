import tensorflow as tf
import numpy as np

class ModelParameters(object):
    """Parameters for the modeled system

    Parameters
    ---------------

    """
    def __init__(   self, nsamples = 1e4, model_type = "rep", nspecies = 3,
                    n0 = 6, k0 = 0.5, dt0 = 0.1, t_end = 20):
        # TODO: add checks, assertions

        # define values of constant parameters
        nsteps = int(t_end / dt0)
        self.nsteps = nsteps
        self.nspecies = nspecies
        self.model_type = model_type
        self.nsamples = int(nsamples)
        gamma0 = 1/(1+np.power(k0, n0))
        self.params = ( tf.constant(n0, shape = [],
                                    dtype = tf.float32,   name = "n"),
                        tf.constant(k0, shape = [],
                                    dtype = tf.float32, name = "k"),
                        tf.constant(gamma0,shape=[],
                                    dtype = tf.float32, name = "gamma"),
                        tf.constant(dt0,shape = [],
                                    dtype = tf.float32, name = "dt"))

class Simulator(ModelParameters):
    def __init__(self, model_params, x_low = 0.2, seed = 123, rv_std = 0.1):

        self.mp = model_params

        ## Repressilator
        if self.mp.model_type.startswith("r"):
            x0 = np.ones((self.mp.nsamples,  self.mp.nspecies), dtype = "float32")
            # Start with one high (=1) and rest low
            x0[:, 1:self.mp.nspecies] = x_low

        ## Genetic Switch
        elif self.mp.model_type.startswith("s"):
            x0 = 0.5*np.ones((self.mp.nsamples, 1), dtype= "float32")
            self.mp.nspecies = 1

        # if done like this, must feed_dict on Session.run
        # x0 = tf.placeholder(tf.float32, name = "x0", shape = init.shape)
        x0 = tf.convert_to_tensor(x0, dtype = tf.float32, name = "x0")
        self.x0 = x0

        self.rv_n = tf.random_normal(shape = \
                            [self.mp.nsteps, self.mp.nsamples, self.mp.nspecies],
                            stddev = rv_std, seed = seed, name = "rv_n")

    def evolve(self, x, n, k, gamma):
        """ Compute time-derivative at current state

        Model: dx/dt = x^n / (x^n + K^n) - gamma*x
        This leads to single-species bistability.
        """
        dxdt = tf.pow(x, n)/(tf.pow(x, n)+tf.pow(k,n)) - gamma*x
        return dxdt

    def evolve_system(self, x, n, k, gamma):
        """ Compute time-derivative at current state

        Model: dx/dt = k^n / (x^n + K^n) - gamma*x
        This leads to 3+ species sustained oscillations. Note that x is matrix.

        We have dependency only on preceding variable, which can be efficiently implemented
        by rolling the matrix by `shift=-1` along corresponding axis.
        """
        temp = tf.pow(k, n)/(tf.pow(x, n)+tf.pow(k,n))
        # dxdt = tf.manip.roll(temp, shift = -1, axis = 1) - gamma*x # v1.6+
        dxdt = tf.concat([  tf.reshape(temp[:, -1], [-1, 1]),
                            temp[:,:-1]], axis=1) - gamma*x # v1.5
        dxdt = tf.convert_to_tensor(dxdt, dtype = tf.float32, name = "dxdt")
        return dxdt

    def rk4_sde(self, x, rv_n):
        """Runge-Kutta 4th order method for SDE

        @reference http://people.sc.fsu.edu/~jburkardt/c_src/stochastic_rk/stochastic_rk.html
        @reference `https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)`
        """
        a21 =   2.71644396264860
        a31 = - 6.95653259006152
        a32 =   0.78313689457981
        a41 =   0.0
        a42 =   0.48257353309214
        a43 =   0.26171080165848
        a51 =   0.47012396888046
        a52 =   0.36597075368373
        a53 =   0.08906615686702
        a54 =   0.07483912056879

        q1 =    2.12709852335625
        q2 =    2.73245878238737
        q3 =   11.22760917474960
        q4 =   13.36199560336697

        n = self.mp.params[0]; k = self.mp.params[1];
        gamma = self.mp.params[2]; dt = self.mp.params[3];

        if x.get_shape()[1] > 1:
            evolve_fun = self.evolve_system
        else:
            evolve_fun = self.evolve

        x1 = x
        k1 = dt * evolve_fun(x1, n, k, gamma) + tf.sqrt(dt) * x * rv_n

        x2 = x1 + a21 * k1
        k2 = dt * evolve_fun(x2, n, k, gamma) + tf.sqrt(dt) * x * rv_n

        x3 = x1 + a31 * k1 + a32 * k2
        k3 = dt * evolve_fun(x3, n, k, gamma) + tf.sqrt(dt) * x * rv_n

        x4 = x1 + a41 * k1 + a42 * k2
        k4 = dt * evolve_fun(x4, n, k, gamma) + tf.sqrt(dt) * x * rv_n

        x_new = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4

        return tf.cast(x_new, tf.float32)

    def euler_sde(self, x, rv_n):
        """ Euler-Maruyama method for approximate numerical solution of SDE

        SDE: dX = A(X,t)*dt + B(X,t)*dW,
        => A-drift term, B-diffusion term,
        => dW-Wiener process with zero mean and variance dt

        Euler-Maruyama method for  of SDE
        X(t+1)-X(t) =  A(X,t)dt + B(X,t)*sqrt(dt)*eta,
        => eta-standard normal RV

        @reference https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        """
        n = self.mp.params[0]; k = self.mp.params[1];
        gamma = self.mp.params[2]; dt = self.mp.params[3];

        if x.get_shape()[1] > 1:
            evolve_fun = self.evolve_system
        else:
            evolve_fun = self.evolve

        dx = dt * self.evolve(x, n, k, gamma)
        x = x + dx + tf.sqrt(dt)*x*rv_n
        return tf.cast(x, tf.float32)
