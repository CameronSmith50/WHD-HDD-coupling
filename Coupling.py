# Code which will couple the within and between-host dynamics using various coupling techniques, which will each be separate classes.

#%% Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import odeint
from copy import deepcopy
from Individual import individual
from Population import population
import time
import random
from tqdm import tqdm

#%% Class for transmission coupling

class transmissionCoupling(object):

    def __init__(self,
        WHD_Dynamics,
        BHD_Dynamics,
        tVec,
        params,
        traits,
        transFun,
        recFun = None,
        virFun = None,
        innoc = None,
        M = 100,
        vol = 1,
        SS = False,
        initGenotype = None,
        genMut = None,
        WSteadyState = None,
        stochasticSim = False,
        stochasticGill = False,
        WHDvol = 1,
        stoichMat = None,
        seed = None):
        '''
        Function to couple a BHD and WHD simulation. Assumes only 1 susceptible, infected and recovered class
        
        INPUTS
        ------
        WHD_Dynamics: Type - dict. A dictionary which contains the inputs for a WHD simulation. The dictionary should contain the following keys:
            WHD_State - State variables as a dict
            WHD_Params - Parameters for WHDs as a dict
            WHD_Traits - Traits for the WHDs as a dict
            WHD_Dynamics - Dynamics for the WH system as a function.
        See the Individual class for more information on these.
        BHD_Dynamics: Type - dict. A dictionary which contains the inputs for a BHD simulation. The dictionary should contain the following in order:
            BHD_State - State variables as a dict
            BHD_Params - Parameters for BHDs as a dict
            BHD_Traits - Traits for the BHDs as a dict
            BHD_Dynamics - Dynamics for the BH system as a function.
        See the Population class for more information. Note that the transmission terms should not appear in the ODEs. Nor should the recovery or virulence terms if specified as an option. There should also be an additional input to the dictionary
            BHD_State_Type: Type - np.ndarray. This should be the same length as the number of BHD state variables, with a 1 indicating the susceptible variable, a 2 indicating an infected state, a 3 indicating the recovered state and 0 indicating any other.
        tVec: type - np.ndarray. An array with the times which should contain the BHDs recording mesh. Should be a constant mesh.
        params: Type - dict. Parameters for the coupling. Should contain the parameters for any of the coupling functions defined.
        traits: Type - dict. Traits for the coupling.
        transFun: Type - function. A function which has inputs:
            S - The state variables from the WHD_dynamics
            P - Parameters for the coupling
            T - Traits for the coupling.
        Should output the propensity function for a given individual to infect a susceptible.
        recFun: Type - function: If none, uses the BHDs for recovery (if applicable). If specified, should be of the same form as transFun, but outputs the propensity function for an individual recovering.
        virFun: Type - function: If none, uses the BHDs for virulence (if applicable). If specified, should be of the same form as transFun, but outputs the propensity function for an individual dying due to disease.
        innoc: Type - function: If none, all new individuals are infected with the same innoculum. If specified, the function gives the initial innoculum based on the individual who infected the. The function should have the same inputs as transFun, but output the initial innoculum.
        M: Type - int. Number of independent repeats.
        vol: Type - float. Volume of the system
        SS: Type - bool. Variable that decides if the WHD steady states should be used for coupling or not
        initGenotype: Type - str. String of 0s and 1s to denote the initial genotype in the population
        genMut: Type - float. The mutation rate for a single element of the genotype.
        WSteadyState: Type - ndarray. Contains the steady state value for the within host dynamics. If unspecified, this is calculated in the code if needed.
        stochasticSim: Type - bool. Are we using a stochastic simulation at the within-host level?
        stochasticGill: Type - bool. If using a stochastic sim, is it Gillespie (set to True) or tau-leaping (set to false).
        WHDvol: Type - float. Volume for stochastic simulations.
        stoichMat: Type - ndarray. Stoicheometric matrix. Rows should correspond to the WHD_ODEs rows.
        '''

        # Set the seed to specified value if given
        if seed != None:
            np.random.seed(seed)
            random.seed(seed)

        # Save the inputs to the class
        self.WHD_Dynamics = WHD_Dynamics
        self.BHD_Dynamics = BHD_Dynamics
        self.tVec = tVec
        self.params = params
        self.traits = traits
        self.transFun = transFun
        self.recFun = recFun
        self.virFun = virFun
        self.innoc = innoc
        self.M = M
        self.vol = vol
        self.SS = SS
        self.initGenotype = initGenotype
        self.genMut = genMut
        self.WSteadyState = WSteadyState
        self.stochasticSim = stochasticSim
        self.stochasticGill = stochasticGill
        self.WHDvol = WHDvol
        self.stoichMat = stoichMat
        if stochasticSim:
            if stoichMat.any() == None:
                return('Stoichiometric matrix not defined')

        # Unpack the WHDs and create an individual which can be copied
        self.WState = WHD_Dynamics['WHD_State']
        self.WNState = len(self.WState)
        self.WParams = WHD_Dynamics['WHD_Params']
        self.WTraits = WHD_Dynamics['WHD_Traits']
        self.WDynamics = WHD_Dynamics['WHD_Dynamics']
        self.baseIndividual = individual(self.WState,
            self.WParams,
            self.WTraits,
            self.WDynamics,
            infGenotype = self.initGenotype,
            genMut = self.genMut,
            WHDvol = self.WHDvol,
            stochasticSim = stochasticSim,
            stochasticGill = stochasticGill,
            stoichMat = stoichMat)

        # Unpack the BHDs and create a population which can be copied
        self.BState = BHD_Dynamics['BHD_State']
        self.BNState = len(self.BState)
        self.BParams = BHD_Dynamics['BHD_Params']
        self.BTraits = BHD_Dynamics['BHD_Traits']
        self.BDynamics = BHD_Dynamics['BHD_Dynamics']
        self.basePopulation = population(self.BState,
            self.BParams,
            self.BTraits,
            self.BDynamics,
            M = self.M,
            vol = self.vol)
        self.BHD_State_Type = BHD_Dynamics['BHD_State_Type']
        self.susStates = ((self.BHD_State_Type == 1*np.ones(self.BNState))*(np.arange(self.BNState)+1))
        self.susStates = (self.susStates[self.susStates > 0] - 1)[0]
        self.infStates = ((self.BHD_State_Type == 2*np.ones(self.BNState))*(np.arange(self.BNState)+1))
        self.infStates = (self.infStates[self.infStates > 0] - 1)[0]
        self.recStates = ((self.BHD_State_Type == 3*np.ones(self.BNState))*(np.arange(self.BNState)+1))
        self.recStates = (self.recStates[self.recStates > 0] - 1)[0]
        self.simulated = False  # Determines if a simulation has been run
        self.N = np.sum(list(self.BState.values()))

        # Create temporal parameters
        self.tFinal = self.tVec[-1]  # Final time of simulation
        self.nt = len(self.tVec)  # Number of time points
        self.dt = self.tVec[1]-self.tVec[0]

        # Add further parameters
        self.parVals = list(self.params.values())
        self.traitVals = list(self.traits.values())

        # Create a storage matrix for the BHDs
        self.BHDs = []
        self.BHDStateStore = np.zeros((self.nt, self.BNState, self.M))

        # Find the WHD steady state if applicable
        if SS and WSteadyState == None:
            II = deepcopy(self.baseIndividual)
            self.steadyState = odeint(lambda X, t: II.RHSDynamics(X, t), II.state, np.linspace(0, self.tFinal*10, 10*(self.nt-1)-1))[-1]
        elif SS and WSteadyState != None and len(WSteadyState) == self.WNState:
            self.steadyState = WSteadyState
        elif SS and WSteadyState != None and len(WSteadyState) != self.WNState:
            print('The length of the specified steady state is not the same as the number of WHD state variables.')
            return

    # Define the transmission function
    def transmission(self, individual):
        '''
        Calculate the transmission that an individual has.
        Must input the class of the individual.
        '''

        if self.SS:
            return(self.transFun(self.steadyState, self.parVals, self.traitVals))
        else:
            return(self.transFun(individual.state, self.parVals, self.traitVals))

    # Define the recovery function
    def recovery(self, individual):
        '''
        Calculate the recovery that an individual has.
        Must input the class of the individual.
        If there is no function as specified by inputs, then returns None.
        '''

        if self.recFun != None:
            if self.SS:
                return(self.recFun(self.steadyState, self.parVals, self.traitVals))
            else:
                return(self.recFun(individual.state, self.parVals, self.traitVals))

        else:
            return(0)

    # Define the virulence function
    def virulence(self, individual):
        '''
        Calculate the virulence that an individual has.
        Must input the class of the individual.
        If there is no function as specified by inputs, then returns None.
        '''

        if self.virFun != None:
            if self.SS:
                return(self.virFun(self.steadyState, self.parVals, self.traitVals))
            else:
                return(self.virFun(individual.state, self.parVals, self.traitVals))

        else:
            return(0)

    def initialise(self, m):

        # Initialise a population level model
        self.pop = deepcopy(self.basePopulation)

        # Initialise the time
        self.t = 0
        self.dtNext = self.dt
        self.dtInd = 1

        # Find the numbers in infective and recovered classes
        self.NI = (np.floor(self.pop.state[self.infStates]*self.vol).astype(int))
        self.NR = (np.floor(self.pop.state[self.recStates]*self.vol).astype(int))

        # Initialise the appropriate number of infected individuals
        self.infList = []
        self.recList = []
        self.remList = []
        for jj in range(self.NI):
            self.infList.append(deepcopy(self.baseIndividual))
            self.infList[-1].index = self.pop.currentInd
            self.infList[-1].infGenotype = self.initGenotype
            self.infList[-1].genotype = self.initGenotype
            self.infList[-1].genMut = self.genMut
            self.pop.currentInd += 1

        if m+1 == self.M and self.initGenotype != None:
            self.timeDict = {}  # Times that each gene appears first
            self.geneDict = {}  # Connections between genes
            self.IDDict = {}  # Gene identifier
            self.nGeneDict = {}  # Number of current individuals with each genotype
            self.nVecDict = {}  # Vector of number of individuals with each genotype
            self.extDict = {}  # Extinction times for each gene
            self.timeDict[self.initGenotype] = [0]
            self.IDDict[self.initGenotype] = 0
            self.nGeneDict[self.initGenotype] = self.NI
            self.nVecDict[self.initGenotype] = [1]
            self.currentID = 1

    # Find the Gillespie time-step
    def GillsepieTime(self, m):

        # t0 = time.perf_counter() 

        # Check if we should remove an infected individual because of the BHDs
        # If so, we need to choose one to go into an appropriate list
        if np.random.rand() < self.NI - self.pop.state[self.infStates]*self.vol:
            
            # Choose a random infected individual
            randInd = np.random.randint(self.NI)

            # If we have cross-scale mortality, then we must have a recovered individual
            if self.virulence(self.baseIndividual) != None:

                self.recList.append(self.infList[randInd])
                self.infList.pop(randInd)
                self.recList[-1].status = 'R'
                self.NI -= 1
                self.NR += 1

                if m+1 == self.M and self.initGenotype != None:
                    self.nGeneDict[self.recList[-1].genotype] -= 1

            # Otherwise, we add them to the removed list
            else:

                self.remList.append(self.pop.infList[randInd])
                self.infList.pop(randInd)

                if self.recovery(self.baseIndividual) != None:
                    self.remList[-1].status = 'D'
                else:
                    self.remList[-1].status = 'N/A'  
                self.NI -= 1

                if m+1 == self.M and self.initGenotype != None:
                    self.nGeneDict[self.remList[-1].genotype] -= 1

        # Check if we should remove a recovered individual because of the BHDs
        if np.random.rand() < self.NR - self.pop.state[self.recStates]*self.vol:

            # Choose a random infected recovered
            randRec = np.random.randint(self.NR)

            # Remove a recovered and place them in removed
            self.remList.append(self.recList[randRec])
            self.recList.pop(randRec)
            self.remList[-1].status = 'D'
            self.NR -= 1
            

        # Firstly we calculate all of our propensity functions.
        # We will have:
        #   self.NI for transmission events
        #   self.NI for recovery events
        #   self.NI for mortality events
        self.props = np.zeros(3*self.NI)
        for ii in range(self.NI):
            self.props[ii] = self.transmission(self.infList[ii])/self.vol*self.pop.state[self.susStates]*self.vol
            self.props[self.NI + ii] = self.recovery(self.infList[ii])
            self.props[2*self.NI + ii] = self.virulence(self.infList[ii])

        # Find the time until next event
        self.sumProps = np.sum(self.props)
        if self.sumProps > 0:
            self.GillTime = 1/self.sumProps*np.log(1/np.random.rand())
        else:
            self.GillTime = np.inf

        # self.GillTimeCount += time.perf_counter() - t0
        # self.nGill += 1

    # A Gillespie simulation step
    def GillespieStep(self, m):
        
        # Find the event to enact
        event = 0
        randNum = self.sumProps*np.random.rand()
        cumsum = self.props[0]
        while cumsum < randNum:
            event += 1
            cumsum += self.props[event]

        # Update time
        self.t += self.GillTime

        if event < self.NI:  # A transmission event from individual event has occurred

            # t0 = time.perf_counter()                    

            # Find the individual transmitting
            indToTrans = event

            # Create a new infected
            newInf = individual(self.WState,
                self.WParams,
                self.WTraits,
                self.WDynamics)
            newInf.status = 'I'
            newInf.BHDTime = self.t
            newInf.index = self.pop.currentInd
            newInf.infector = self.infList[indToTrans].index
            newInf.infGenotype = self.infList[indToTrans].genotype
            newInf.genotype = self.infList[indToTrans].genotype
            newInf.genMut = self.infList[indToTrans].genMut
            self.infList.append(newInf)

            # Write to the infector's infection list
            self.infList[indToTrans].infected.append(self.pop.currentInd)

            # Update state numbers
            self.pop.state[self.susStates] -= 1/self.vol
            self.NI += 1
            self.pop.state[self.infStates] += 1/self.vol

            # Update the index
            self.pop.currentInd += 1

            if m+1 == self.M and self.initGenotype != None:
                self.nGeneDict[newInf.genotype] += 1

            # self.GillTimeCount += time.perf_counter() - t0
            # self.nGill += 1

        elif event < 2*self.NI:  # Infected individual event - self.NI recovers

            # t0 = time.perf_counter() 
            self.RR += 1

            # Find the individual recovering
            indToRec = event - self.NI

            # Convert the infected to a recovered
            newRec = self.infList[indToRec]
            self.infList.pop(indToRec)
            newRec.status = 'R'
            self.recList.append(newRec)

            # Update state numbers
            self.NI -= 1
            self.pop.state[self.infStates] -= 1/self.vol
            self.NR += 1
            self.pop.state[self.recStates] += 1/self.vol

            if m+1 == self.M and self.initGenotype != None:
                self.nGeneDict[newRec.genotype] -= 1

            # Check if self.NI has become 0. If so, set the state to be 0
            if self.NI == 0:
                self.pop.state[self.infStates] = 0

            # self.GillTimeCount += time.perf_counter() - t0
            # self.nGill += 1

        elif event < 3*self.NI:  # Infected individual event - 2*self.NI dies

            # t0 = time.perf_counter() 

            # Find the individual recovering
            indToRem = event - 2*self.NI

            # Convert the infected to a recovered
            newRem = self.infList[indToRem]
            self.infList.pop(indToRem)
            self.remList.append(newRem)

            # Update state numbers
            self.NI -= 1
            self.pop.state[self.infStates] -= 1/self.vol

            if m+1 == self.M and self.initGenotype != None:
                    self.nGeneDict[newRem.genotype] -= 1

            # Check if self.NI has become 0. If so, set the state to be 0
            if self.NI == 0:
                self.pop.state[self.infStates] = 0

            

            # self.GillTimeCount += time.perf_counter() - t0
            # self.nGill += 1

    # A simulation step
    def tauStep(self, dtInd, m):

        tt = time.perf_counter()

        # Update the BHD solution
        stateBef = self.pop.state
        self.pop.updateODEs(self.dt)

        if self.pop.state[self.susStates] < 0:
            self.pop.state[self.susStates] = 0

        # Check if we should remove an infected individual because of the BHDs
        # If so, we need to choose one to go into an appropriate list
        if np.random.rand() < self.NI - self.pop.state[self.infStates]*self.vol:

            # Calculate the number we should remove
            if self.NI - self.pop.state[self.infStates]*self.vol < 1:
                nRandInd = 1
            else:
                nRandInd = np.floor(self.NI - self.pop.state[self.infStates]*self.vol).astype(int)

            # Choose a random infected individual
            randInds = sorted(random.sample(range(self.NI), nRandInd), reverse=True)

            # Loop through the indices
            for randInd in randInds:

                # If we have cross-scale mortality, then we must have a recovered individual
                if self.virulence(self.baseIndividual) != None:

                    self.recList.append(self.infList[randInd])
                    self.infList.pop(randInd)
                    self.recList[-1].status = 'R'
                    self.NI -= 1
                    self.NR += 1

                    # Remove the genotype from the list
                    if m+1 == self.M and self.initGenotype != None:
                        self.nGeneDict[self.recList[-1].genotype] -= 1
                    
                # Otherwise, we add them to the removed list
                else:

                    self.remList.append(self.pop.infList[randInd])
                    self.infList.pop(randInd)

                    if self.recovery(self.baseIndividual) != None:
                        self.pop.remList[-1].status = 'D'
                    else:
                        self.pop.remList[-1].status = 'N/A'  
                    self.NI -= 1

                    # Remove the genotype from the list
                    if m+1 == self.M and self.initGenotype != None:
                        self.nGeneDict[self.recList[-1].genotype] -= 1

        # Check if we should remove a recovered individual because of the BHDs
        if np.random.rand() < self.NR - self.pop.state[self.recStates]*self.vol:

            # Calculate the number we should remove
            if self.NR - self.pop.state[self.recStates]*self.vol < 1:
                nRandRec = 1
            else:
                nRandRec = np.floor(self.NR - self.pop.state[self.recStates]*self.vol).astype(int)

            # Choose a random infected individual
            randRecs = sorted(random.sample(range(self.NR), nRandRec), reverse=True)

            # Loop through the indices
            for randRec in randRecs:

                # Choose a random infected recovered
                randRec = np.random.randint(self.NR)

                # Remove a recovered and place them in removed
                self.remList.append(self.recList[randRec])
                self.recList.pop(randRec)
                self.remList[-1].status = 'D'
                self.NR -= 1

        # Define the propensity functions
        self.props = np.zeros(3*self.NI)
        for ii in range(self.NI):
            self.props[ii] = max(0,self.transmission(self.infList[ii]))*self.pop.state[self.susStates]
            self.props[self.NI + ii] = self.recovery(self.infList[ii])
            self.props[2*self.NI + ii] = self.virulence(self.infList[ii])

        # Find the number of firings that each interaction has
        nFire = np.random.poisson(self.props*self.dt)

        self.tracker.append(self.NI)

        # Temporary storage variables
        IIndsToRem = []
        NSToRem = 0
        NIToAdd = 0
        IIndsToAdd = []
        NRToAdd = 0
        RIndsToAdd = []
        remIndsToAdd = []

        # Transmission
        for ii in range(self.NI):
            nf = nFire[ii]
            for n in range(nf):
                newInf = deepcopy(self.baseIndividual)
                newInf.infector = self.infList[ii].index
                newInf.index = self.pop.currentInd
                self.infList[ii].infected.append(self.pop.currentInd)
                self.infList[ii].tInfected.append(dtInd*self.dt)
                newInf.BHDtime = dtInd*self.dt
                newInf.infGenotype = self.infList[ii].genotype
                newInf.genotype = self.infList[ii].genotype
                newInf.genMut = self.infList[ii].genMut
                newInf.mutMat = self.infList[ii].mutMat
                newInf.mutDist = self.infList[ii].mutDist
                self.pop.currentInd += 1
                NSToRem += 1
                NIToAdd += 1
                IIndsToAdd.append(newInf)

                if m+1 == self.M and self.initGenotype != None:
                    self.nGeneDict[newInf.genotype] += 1

        # Recovery
        for ii in range(self.NI):
            if nFire[self.NI + ii] > 0:
                newRec = deepcopy(self.infList[ii])
                newRec.status = 'R'
                IIndsToRem.append(ii)
                NRToAdd += 1
                RIndsToAdd.append(newRec)

                if m+1 == self.M and self.initGenotype != None:
                    self.nGeneDict[newRec.genotype] -= 1

        # Mortality
        for ii in range(self.NI):
            if nFire[2*self.NI + ii] > 0:
                newRem = deepcopy(self.infList[ii])
                IIndsToRem.append(ii)
                remIndsToAdd.append(newRem)

                if m+1 == self.M and self.initGenotype != None:
                    self.nGeneDict[newRem.genotype] -= 1

        # Resolve each of the lists and counts
        # Sort the index lists to remove, and eliminate duplicates
        IIndsToRem = sorted(list(dict.fromkeys(IIndsToRem)), reverse=True)

        # Remove these individuals
        for ind in IIndsToRem:
            self.infList.pop(ind)

        # Add the new individuals
        for newInd in IIndsToAdd:
            self.infList.append(newInd)
        for newInd in RIndsToAdd:
            self.recList.append(newInd)
        for newInd in remIndsToAdd:
            self.remList.append(newInd)

        # Update state variables
        self.NI = self.NI + NIToAdd - len(IIndsToRem)
        self.NR = self.NR + NRToAdd 
        self.pop.state[self.susStates] -= NSToRem/self.vol
        self.pop.state[self.infStates] += (NIToAdd - len(IIndsToRem))/self.vol
        self.pop.state[self.recStates] += (NRToAdd)/self.vol

        # Check if self.NI has become 0. If so, set the state to be 0
        if self.NI == 0:
            self.pop.state[self.infStates] = 0
        if self.pop.state[self.susStates] < 0:
            self.pop.state[self.susStates] = 0

    def storage(self, dtind, m):
        '''
        Code to store the current state into the storage array
        '''

        self.BHDStateStore[dtind, :, m] = self.pop.state

    def endOfRep(self):
        '''
        Code to store the BHDs for a repeat
        '''
        self.pop.infList = self.infList
        self.pop.remList = self.remList
        self.pop.recList = self.recList
        self.BHDs.append(self.pop)
        

    # Gillespie simulation
    def GillespieSim(self):
        '''
        Code to run a full Gillespie simulation.

        Gillespie wrong here? Time varying version.
        '''

        # Initialise the timing vector
        self.simTimeVec = np.zeros(self.M)
        self.nGill = 0
        self.nStore = 0
        self.GillTimeCount = 0
        self.NN = 0
        self.RR = 0

        # Loop through repeats
        for m in range(self.M):

            # Inistialise the repeat
            self.initialise(m)
            self.storage(0, m)

            # Start the timer
            timeInit = time.perf_counter()

            # Loop through time
            while self.t < self.tFinal:

                # Calculate the time until the next Gillespie step
                self.GillsepieTime(m)

                # If the Gillespie time occurs before the next recording time
                if self.t + self.GillTime < self.dtNext:

                    # Update BHDs
                    # self.pop.updateODEs(self.GillTime)

                    # Complete a Gillespie step
                    self.GillespieStep(m)

                    # Update the WHDs
                    for ii in range(self.NI):
                        t0 = time.perf_counter()
                        self.infList[ii].updateSystem(self.GillTime)
                    # for jj in range(self.NR):
                    #     self.pop.recList[jj].updateSystem(self.GillTime)

                        self.GillTimeCount += time.perf_counter() - t0
                        self.nGill += 1
                
                # Otherwise we have a recording step
                else:

                    # t0 = time.perf_counter()

                    # Find the time step
                    tStep = self.dtNext - self.t

                    # Update time
                    self.t = self.dtNext

                    # Update the BHDs
                    # self.pop.updateODEs(tStep)
                    self.pop.updateODEs(self.dt)

                    # Store the state variable
                    self.storage(self.dtInd, m)

                    # Update each individual
                    for ii in range(self.NI):

                        # Update the ODE system for this individual
                        self.infList[ii].updateSystem(tStep)

                        # Update the genotype if applicable
                        if m+1 == self.M and self.initGenotype != None:

                            # Mutate
                            prevGeno, newGeno = self.infList[ii].mutateGenotype(self.dt)

                            # Check if that gene has been seen yet
                            if self.timeDict.get(newGeno) == None:

                                # If not, initialise its time and mutator
                                self.timeDict[newGeno] = [self.t]

                                # Assign an ID
                                self.IDDict[newGeno] = self.currentID
                                self.currentID += 1

                                # Create a new ntry to track the number of individals with that genotype
                                self.nGeneDict[newGeno] = 1

                            else:
                                self.timeDict[newGeno].append(self.t)
                                self.nGeneDict[newGeno] += 1

                            # Check if this muation has occurred
                            if prevGeno != newGeno:
                                if self.geneDict.get(prevGeno + '-' + newGeno) == None and self.timeDict.get(newGeno) == None:

                                    # If not initilise
                                    self.geneDict[prevGeno + '-' + newGeno] = [self.t]

                                else:
                                    self.geneDict[prevGeno + '-' + newGeno].append(self.t)

                            # Remove 1 from the count of the mutator
                            self.nGeneDict[prevGeno] -= 1

                    # for jj in range(self.NR):
                    #     self.pop.recList[jj].updateSystem(tStep)

                    # Update recording indices
                    self.dtInd += 1
                    if self.dtInd < len(self.tVec):
                        self.dtNext = self.tVec[self.dtInd]

                    self.nStore += 1

                    # self.GillTimeCount += time.perf_counter() - t0
                    # self.nGill += 1

                self.NN += self.NR

            # End the repeat
            self.endOfRep()

            # Find the time taken for the repeat
            timeFinal = time.perf_counter()
            self.simTimeVec[m] = timeFinal - timeInit

        # Set the Boolean for whether the simulation has occurred as true
        self.simulated = True

    def tauSimulation(self):
        '''
        Code to run a simulation.
        '''

        # Initialise the timing vector
        self.simTimeVec = np.zeros(self.M)

        # Loop through repeats
        for m in tqdm(range(self.M), leave = False):

            # print(m)

            # Initialise the repeat
            self.initialise(m)
            self.storage(0, m)

            # Start the timer
            timeInit = time.perf_counter()
            self.timeVec = []
            self.tracker = []

            # Loop through time
            for kk in range(1, self.nt):

                # Conduct a simulation step
                self.tauStep(kk, m)

                # Store the demographics
                self.storage(kk, m)

                tt = time.perf_counter()

                # Update the individuals
                for ii in range(self.NI):
                    
                    self.infList[ii].updateSystem(self.dt)
                    
                    # Update the genotype if applicable
                    if m+1 == self.M and self.initGenotype != None:

                        # Attempt to mutate the individual
                        prevGeno, newGeno = self.infList[ii].mutateGenotype(self.dt)

                        # Has a mutation occurred?
                        if prevGeno != newGeno:

                            # Check if we have seen the new genotype before
                            if self.IDDict.get(newGeno) == None:

                                # If not, add it to the ID dictionary and increment. Initialise the number to be 1
                                self.IDDict[newGeno] = self.currentID
                                self.currentID += 1
                                self.nGeneDict[newGeno] = 1

                            else:

                                self.nGeneDict[newGeno] += 1

                            # Remove one individual from the previous genotyoe
                            self.nGeneDict[prevGeno] -= 1

                            # Now we add a branch under certain conditions
                            # Currently, we will only save a branching event if the new genotype is new
                            if self.geneDict.get(prevGeno + '-' + newGeno) == None and self.timeDict.get(newGeno) == None:

                                # Add to the branch dictionary
                                self.geneDict[prevGeno + '-' + newGeno] = [self.dt*kk]
                
                            # If the branch doesn't exist but we have seen the genotype before
                            elif self.geneDict.get(prevGeno + '-' + newGeno) == None and self.timeDict.get(newGeno) != None:

                                # Do nothing
                                pass

                            # Otherwise, the branch has already happened, so append this new time
                            else:

                                self.geneDict[prevGeno + '-' + newGeno].append(self.dt*kk)
                
                # Now we update each time vector and number of individuals
                if m+1 == self.M and self.initGenotype != None:
                    for key, _ in self.IDDict.items():

                        # If they still exist, add the number of individuals and the current time to appropriate dictionaries
                        if self.nGeneDict[key] > 0:

                            # Update numbers
                            if self.nVecDict.get(key) == None:
                                self.nVecDict[key] = [self.nGeneDict[key]]
                                self.timeDict[key] = [self.dt*kk]
                            else:
                                self.nVecDict[key].append(self.nGeneDict[key])
                                self.timeDict[key].append(self.dt*kk)


                # for jj in range(self.NR):
                #     self.pop.recList[jj].updateSystem(self.dt)

                self.timeVec.append(time.perf_counter()-tt)

            # End the repeat    
            self.endOfRep()

            # Find the time taken for the repeat
            timeFinal = time.perf_counter()
            self.simTimeVec[m] = timeFinal - timeInit

        # Set the simulated Boolean to true
        self.simulated = True

    # Plotting function for BHDs
    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for ii in range(self.BNState):
            ax.fill_between(self.tVec, np.mean(self.BHDStateStore[:, ii, :], axis=1) + np.sqrt(np.var(self.BHDStateStore[:, ii, :], axis=1))/np.sqrt(self.M), np.mean(self.BHDStateStore[:, ii, :], axis=1) - np.sqrt(np.var(self.BHDStateStore[:, ii, :], axis=1))/np.sqrt(self.M), alpha=0.5)
            ax.plot(self.tVec, np.mean(self.BHDStateStore[:, ii, :], axis=1), lw=2, label = list(self.BState.keys())[ii])
        plt.plot(self.tVec, np.sum(np.mean(self.BHDStateStore[:, :, :], axis=2), axis=1), 'k--', lw=2)

        yInit = sum(list(self.BState.values()))
        ax.set_xlim([0, self.tFinal])
        ax.set_ylim([0, yInit/self.vol])

        plt.legend()

    def estR0fromInds(self):
        '''
        Code to estimate the basic reproductive number from the initial individual(s)
        This code can only be run once a simulation has been run.
        '''

        if not self.simulated:

            print('Code has not been executed, so R0 cannot be estimated.')
            return
        
        else:

            # Locate the individual(s) who were initially infected
            nInit = np.floor(np.array(list(self.BState.values()))[self.infStates]).astype(int)

            # Create an empty array
            nSec = np.zeros((nInit, self.M))

            # Create a temporary list
            tempList = []

            # Loop through the repeats
            for m in range(self.M):

                # Extract the BHDs for that repeat
                sim = self.BHDs[m]

                # Loop through the initial indices
                for ii in range(nInit):

                    # Look in each of the storage lists
                    # We add 1 in each of the below because counting starts at 1
                    for jj in range(len(sim.infList)):
                        if sim.infList[jj].index == ii + 1:
                            tempList.append(sim.infList[jj])

                    for jj in range(len(sim.recList)):
                        if sim.recList[jj].index == ii + 1:
                            tempList.append(sim.recList[jj])

                    for jj in range(len(sim.remList)):
                        if sim.remList[jj].index == ii + 1:
                            tempList.append(sim.remList[jj])

                # Now take each of these and count the number of secondary infections they caused
                for kk in range(nInit):

                    nSec[kk, m] = len(tempList[kk + m*nInit].infected)

        # print('Estimate for R0 from the individuals is ' + str(np.mean(nSec)) + ', with a standard error of ' + str(np.sqrt(np.var(nSec)/self.M)))

        # counts, bins = np.histogram(np.reshape(nSec, nInit*self.M), bins=np.linspace(-0.5, 15.5, 17), density=True)
        # plt.hist(bins[:-1], bins, weights=counts)
        # plt.show()

        aveVal = np.mean(nSec)
        varVal = np.sqrt(np.var(nSec))

        return(aveVal, varVal)

    def estR0fromFun(self):
        '''
        Code to estimate the value of R0 from the integral form
        '''

        # Create a test individual
        testIndividual = deepcopy(self.baseIndividual)

        # Create storage vectors and initial conditions
        surv = [1]
        aveSec = [0]
        varSec = [0]
        prevSurv = 1
        prevAveSec = 0
        prevVarSec = 0
        S0 = list(self.BState.values())[self.susStates]
        transList = [self.transmission(testIndividual)]
        
        # Loop through the time steps
        for ii in range(1, self.nt):

            # Update the WHD solution
            testIndividual.updateSystem(self.dt)

            # Update the survivorship function
            newSurv = prevSurv/(1 + (self.recovery(testIndividual) + self.virulence(testIndividual))*self.dt)
            surv.append(newSurv)
            prevSurv = newSurv

            # Update the average secondaries list
            trans = self.transmission(testIndividual)/self.vol
            transList.append(trans)
            newAveSec = root(lambda X: X - trans*surv[-1]*(S0 - X)*self.dt - prevAveSec, prevAveSec).x[0]
            aveSec.append(newAveSec)
            prevAveSec = newAveSec

            # Update the variance list
            newVarSec = root(lambda X: X - newSurv*trans*(S0 - newAveSec - 2*X)*self.dt - prevVarSec, prevVarSec).x
            varSec.append(newVarSec)
            prevVarSec = newVarSec
        
        # Calculate R0
        ave = aveSec[-1]
        var = varSec[-1]

        integrandVec = np.array(surv)*np.array(transList)*S0

        R0est = 0.5*sum(integrandVec[:-1] + integrandVec[1:])*self.dt

        # print('Estimate for R0 from the function form is ' + str(ave) + ' or ' + str(R0est))
        # print(str(S0*((1-np.exp(-self.params['beta']*self.WParams['K']/(self.params['recov']+self.params['b']))))))

        # fig = plt.figure()
        # plt.plot(self.tVec, aveSec)
        # plt.plot(self.tVec, transList)
        # plt.show()

        return(ave, var)

    def createHistFromR0(self, plot=True):
        '''
        Creates a histogram from R0 data
        '''

        if not self.simulated:

            print('Code has not been executed, so R0 cannot be estimated.')
            return
        
        else:

            # Locate the individual(s) who were initially infected
            nInit = np.floor(np.array(list(self.BState.values()))[self.infStates]).astype(int)

            # Create an empty array
            nSec = np.zeros((nInit, self.M), dtype=int)

            # Create a temporary list
            tempList = []

            # Loop through the repeats
            for m in range(self.M):

                # Extract the BHDs for that repeat
                sim = self.BHDs[m]

                # Loop through the initial indices
                for ii in range(nInit):

                    # Look in each of the storage lists
                    # We add 1 in each of the below because counting starts at 1
                    for jj in range(len(sim.infList)):
                        if sim.infList[jj].index == ii + 1:
                            tempList.append(sim.infList[jj])

                    for jj in range(len(sim.recList)):
                        if sim.recList[jj].index == ii + 1:
                            tempList.append(sim.recList[jj])

                    for jj in range(len(sim.remList)):
                        if sim.remList[jj].index == ii + 1:
                            tempList.append(sim.remList[jj])

                # Now take each of these and count the number of secondary infections they caused
                for kk in range(nInit):

                    nSec[kk, m] = len(tempList[kk + m*nInit].infected)
        
        # We bin our data
        counts, bins = np.histogram(nSec, bins=np.arange(-0.5, 21, 1))

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(bins[1:]-0.5, counts/np.sum(counts))
            ax.set_xlim(-0.5,20.5)
            ax.set_xticks(np.arange(0,22,2))

        return(counts, bins)

    def createR0Vec(self):
        '''
        Creates a histogram from R0 data
        '''

        if not self.simulated:

            print('Code has not been executed, so R0 cannot be estimated.')
            return
        
        else:

            # Locate the individual(s) who were initially infected
            nInit = np.floor(np.array(list(self.BState.values()))[self.infStates]).astype(int)

            # Create an empty array
            nSec = np.zeros((nInit, self.M), dtype=int)

            # Create a temporary list
            tempList = []

            # Loop through the repeats
            for m in range(self.M):

                # Extract the BHDs for that repeat
                sim = self.BHDs[m]

                # Loop through the initial indices
                for ii in range(nInit):

                    # Look in each of the storage lists
                    # We add 1 in each of the below because counting starts at 1
                    for jj in range(len(sim.infList)):
                        if sim.infList[jj].index == ii + 1:
                            tempList.append(sim.infList[jj])

                    for jj in range(len(sim.recList)):
                        if sim.recList[jj].index == ii + 1:
                            tempList.append(sim.recList[jj])

                    for jj in range(len(sim.remList)):
                        if sim.remList[jj].index == ii + 1:
                            tempList.append(sim.remList[jj])

                # Now take each of these and count the number of secondary infections they caused
                for kk in range(nInit):

                    nSec[kk, m] = len(tempList[kk + m*nInit].infected)

        return(nSec)


