import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

class ModelParameters:
    """Parameters for the modeled system

    Parameters
    ---------------

    """
    def __init__(   self, nsamples = 1e4, model_type = "rep", nspecies = 3,
                    x_low = 0.2, n0 = 6, k0 = 0.5, dt0 = 0.1, t_end = 20):
        # TODO: add checks, assertions

        # define output symbolic variable and initial condition based on model
        ## Represilator
        if model_type.startswith("rep"):
            x = T.fmatrix("x")
            init = np.ones((int(nsamples), nspecies), dtype = "float32")
            # Start with one high (=1) and rest low
            init[:, 1:nspecies] = x_low
            x0 = theano.shared(init)

        ## Genetic Switch
        elif model_type.startswith("sw") and nspecies == 2:
            x = T.fmatrix("x")
            x0 = theano.shared(k0*np.ones((int(nsamples), nspecies), dtype='float32'))

        ## Genetic Switch Simplified
        elif model_type.startswith("sw"):
            x = T.fvector("x")
            x0 = theano.shared(k0*np.ones(int(nsamples),dtype='float32'))

        self.x0 = x0
        self.x = x

        #define symbolic placeholders for constant parameters
        dt = T.fscalar("dt")
        k = T.fscalar("k")
        gamma = T.fscalar("gamma")
        n = T.fscalar("n")
        self.params = (n, k , gamma, dt)

        # define values of constant parameters
        gamma0 = 1/(1+np.power(k0, n0))
        self.params0 = (n0, k0, gamma0, dt0)

        self.nsteps = int(t_end / dt0)

class Simulator():
    def __init__(self, seed = 123, rv_shape = (1e4, ), rv_std = 0.1):
        self.rng = RandomStreams(seed)
        self.rv_n = self.rng.normal(rv_shape, std = rv_std)

    def evolve(self, x, n, k, gamma):
        """ Compute time-derivative at current state

        Model: dx/dt = x^n / (x^n + K^n) - gamma*x
        This leads to single-species bistability.
        """
        dxdt = T.pow(x, n)/(T.pow(x, n)+T.pow(k,n)) - gamma*x
        return dxdt

    def evolve_system(self, x, n, k, gamma):
        """ Compute time-derivative at current state

        Model: dx/dt = k^n / (x^n + K^n) - gamma*x
        This leads to 3+ species sustained oscillations. Note that x is matrix.

        We have dependency only on preceding variable, which can be efficiently implemented
        by rolling the matrix by `shift=-1` along corresponding axis.
        """
        temp = T.pow(k, n)/(T.pow(x, n)+T.pow(k,n))
        dxdt = T.roll(temp, shift = -1, axis = 1) - gamma*x
        return dxdt

    def rk4_sde(self, x, n, k, gamma, dt):
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

        if x.ndim > 1:
            evolve_fun = self.evolve_system
        else:
            evolve_fun = self.evolve

        x1 = x
        k1 = dt * evolve_fun(x1, n, k, gamma) + T.sqrt(dt) * x * self.rv_n

        x2 = x1 + a21 * k1
        k2 = dt * evolve_fun(x2, n, k, gamma) + T.sqrt(dt) * x * self.rv_n

        x3 = x1 + a31 * k1 + a32 * k2
        k3 = dt * evolve_fun(x3, n, k, gamma) + T.sqrt(dt) * x * self.rv_n

        x4 = x1 + a41 * k1 + a42 * k2
        k4 = dt * evolve_fun(x4, n, k, gamma) + T.sqrt(dt) * x * self.rv_n

        x_new = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4

        return T.cast(x_new, 'float32')

    def euler_sde(self, x, n, k, gamma, dt):
        """ Euler-Maruyama method for approximate numerical solution of SDE

        SDE: dX = A(X,t)*dt + B(X,t)*dW,
        => A-drift term, B-diffusion term,
        => dW-Wiener process with zero mean and variance dt

        Euler-Maruyama method for  of SDE
        X(t+1)-X(t) =  A(X,t)dt + B(X,t)*sqrt(dt)*eta,
        => eta-standard normal RV

        @reference https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        """
        if x.ndim > 1:
            evolve_fun = self.evolve_system
        else:
            evolve_fun = self.evolve

        dx = dt*evolve_fun(x, n, k, gamma)
        x = x + dx + T.sqrt(dt)*x*self.rv_n
        return T.cast(x, 'float32')
