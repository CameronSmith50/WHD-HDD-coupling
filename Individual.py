# Python code containing a class which stores the information for a given individual. The user is required to specify state variables, parameters, traits and within-host dynamics ODEs.

#%% Required modules and packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from copy import deepcopy

#%% Class

class individual(object):

    def __init__(self,
        stateVars,
        params,
        traits,
        WHD_ODEs,
        infTime = 0,
        index = 0,
        infector = 0,
        infGenotype = None,
        genMut = None,
        mutMat = None,
        mutDist = None,
        stochasticSim = False,
        stochasticGill = False,
        WHDvol = 1,
        stoichMat = None):
        '''
        Initialise an individual for the WHDs.
        
        INPUT
        -----
        stateVars: Type - dict. The state variables of the WHDs with the keys being the variable name, and the     value being the initial value for that variable.
        params: Type - dict. The parameters of the system. Keys are the parameter names and values are their value.
        traits: Type - dict. Traits of the system. Keys are the trait names and values are their value.
        WHD_ODEs: Type - function. A function that defines the RHS of the WHD ODEs or the propensity functions. The function should have inputs (in this order) of state variable, parameters, traits, which should all be vector floats in the order specified in the corresponding dictionaries
        infTime: Type - float. BHD time of infection.
        index: Type - int. Index of the infected individual.
        infector: Type - int. Index of the individual who infected.
        infGenotype: Type - str. The infecting genotype. A string of 0s and 1s.
        genMut: Type - float. The mutation rate for a single element of the genotype.
        mutMat: Type - ndarray. Matri showing transition probabilities between different values at a genotype site.
        mutDist: Type - ndarray. Distribution showing probabilit of each site mutating.
        stochasticSim: Type - bool. Are we using a stochastic simulation at the within-host level?
        stochasticGill: Type - bool. If using a stochastic sim, is it Gillespie (set to True) or tau-leaping (set to false).
        WHDvol: Type - float. Volume for stochastic simulations.
        stoichMat: Type - ndarray. Stoicheometric matrix. Rows should correspond to the WHD_ODEs rows.
        '''

        # Add any inputs to the class
        self.stateVars = stateVars
        self.params = params
        self.traits = traits
        self.WHD_ODEs = WHD_ODEs
        self.infTime = infTime
        self.index = index
        self.infector = infector
        self.infGenotype = infGenotype
        self.genMut = genMut
        self.mutMat = mutMat
        self.mutDist = mutDist
        self.stochasticSim = stochasticSim
        self.stochasticGill = stochasticGill
        self.WHDvol = WHDvol
        self.stoichMat = stoichMat
        if stochasticSim:
            if stoichMat.any() == None:
                return('Stoichiometric matrix not defined')

        # Add further variables
        self.infTime = 0  # Time elapsed since infection
        self.BHDtime = 0  # BHD of infection
        self.state = np.array(list(stateVars.values()))/WHDvol  # The initial state
        self.parVals = list(params.values())  # The parameter values
        self.traitVals = list(traits.values())  # The trait values
        self.infected = []  # List containing indices that this individual has infected
        self.tInfected = []
        self.status = 'I'
        self.traitInds = []
        self.traitVals = []
        for keys, vals in traits.items():
            self.traitInds.append(vals[0])
            self.traitVals.append(vals[1])

        # Genotype parameters
        self.genotype = deepcopy(self.infGenotype)  # Current genotype
        if self.infGenotype != None and self.mutDist == None:
            self.mutDist = 1/len(self.genotype)*np.ones(len(self.genotype))
        if self.infGenotype != None and self.mutMat == None:
            self.mutMat = 1/3*np.ones((4,4)) - 1/3*np.identity(4)

        # Storage lists
        self.tStore = [self.infTime]
        self.stateStore = [self.state]

    def __str__(self):
        return('Individual ' + str(self.index) + ' with traits ' + str(self.traitVals))

    def RHSDynamics(self, state, t):
        '''
        Dynamics for progressing the WHDs
        '''

        return(np.array(self.WHD_ODEs(state, self.parVals, self.traitVals, t)))

    def updateSystem(self, dt):
        '''
        Method to update the dynamics depending on the inputs of the class instance
        If stochasticSim is False, use implicit Euler.
        If stochasticSim is True and stochasticGill is False, use tau leaping.
        If stochasticSim is True and stochasticGill is True, use the Gillespie algorithm.
        '''

        # If using ODEs
        if not self.stochasticSim:

            # Update the state
            # self.state = root(lambda X: X - self.RHSDynamics(X, self.infTime)*dt - self.state, self.state).x
            # self.state = self.state + self.RHSDynamics(self.state, self.infTime)*dt
            k1 = dt*self.RHSDynamics(self.state, self.infTime)
            k2 = dt*self.RHSDynamics(self.state + k1/2, self.infTime + dt/2)
            k3 = dt*self.RHSDynamics(self.state + k2/2, self.infTime + dt/2)
            k4 = dt*self.RHSDynamics(self.state + k3, self.infTime + dt)
            self.state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6

        # Tau-leaping
        elif self.stochasticSim and (not self.stochasticGill):

            # Calculate the number of events in the time-interval of each
            try:
                nFire = np.random.poisson(self.RHSDynamics(self.state*self.WHDvol, self.infTime)*dt)
            except:
                nFire = np.round(np.random.normal(self.RHSDynamics(self.state*self.WHDvol, self.infTime)*dt, np.sqrt(self.RHSDynamics(self.state*self.WHDvol, self.infTime)*dt)))

            # For each one, use the stoicheometric matrix to update the state
            for ii in range(len(nFire)):
                self.state = self.state + nFire[ii]*self.stoichMat[ii,:]/self.WHDvol

            # Look at each state variable and ensure they are all non negative
            for jj in range(len(self.state)):
                self.state[jj] = np.maximum(self.state[jj], 0)

        else:

            # Temporary time
            tTemp = 0

            # Calculate the time until next firing
            props = np.array(self.WHD_ODEs(self.state*self.WHDvol, self.parVals, self.traitVals, self.infTime))
            sumProps = np.sum(props)
            tau = 1/sumProps*np.log(1/np.random.rand())
            tTemp += tau

            # Loop through time
            while tTemp < dt:

                # Find the event that fired
                event = 0
                sumRand = sumProps*np.random.rand()
                cumsum = props[0]
                while cumsum < sumRand:
                    event += 1
                    cumsum += props[event]

                # Enact the event
                self.state = self.state + self.stoichMat[event,:]/self.WHDvol

                # Update propensities
                props = np.array(self.WHD_ODEs(self.state*self.WHDvol, self.parVals, self.traitVals, self.infTime))
                sumProps = np.sum(props)
                tau = 1/sumProps*np.log(1/np.random.rand())
                tTemp += tau


        # Update the storage values
        self.infTime += dt
        self.tStore.append(self.infTime)
        self.stateStore.append(self.state)

    def mutateGenotype(self, dt):
        '''
        Code to mutate the genotype over a time interval dt. Assumes 4 possible values at each site.
        '''

        # Find the length of the genotype
        lenGen = len(self.genotype)
        prevGeno = deepcopy(self.genotype)

        # Calculate the number that should be mutated
        nMut = np.random.poisson(self.genMut*lenGen*dt)
        randGene = np.random.permutation(np.arange(lenGen))
        for ii in randGene[:nMut]:
                
                # Place elements in list
                elements = np.array([int(jj) for jj in self.genotype])

                # Mutate to a new value
                newVal = np.random.choice(4, 1, p=self.mutMat[elements[ii],])[0]

                # Update
                elements[ii] = newVal
                self.genotype = ''.join([str(jj) for jj in elements])        
        
        return(prevGeno, self.genotype)

    def plot(self):
        '''
        Tool to plot the dynamics of the individual
        '''

        # Extract the state variables
        keys = list(self.stateVars.keys())

        # Create the figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # Loop through the state variables and plot
        for ii in range(len(keys)):
            plt.plot(self.tStore, np.array(self.stateStore)[:,ii], lw=2, label=keys[ii])

        # Add a legend
        ax.legend()
        plt.show()

if __name__ == '__main__':

    # We will use logistic growth with intrinsic rate r and carrying capacity K.
    r = 5
    K = 1e10

    # Initialise an individual
    indiv = individual(
        stateVars = {'P': 1/K},
        params = {'r': r},
        traits = {},
        WHD_ODEs = lambda S, P, T, t: S[0]*P[0]*(1-S[0])
    )

    # Simulate their WHDs over ten days and plot
    for ii in range(1000):
        indiv.updateSystem(0.01)
    indiv.plot()
    plt.show()
    
        

    