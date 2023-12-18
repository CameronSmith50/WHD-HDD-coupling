# Python code containing a class which runs the between-host dynmaics. The user is required to specify state variables, parameters, traits and between-host dynamics ODEs.

#%% Required modules and packages

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import root

#%% Class

class population(object):

    def __init__(self,
        stateVars,
        params,
        traits,
        BHD_ODEs,
        M = 100,
        vol = 1):
        '''
        Initialise the BHDs.
        
        INPUT
        -----
        stateVars: Type - dict. The state variables of the BHDs with the keys being the variable name, and the value being the initial value for that variable.
        params: Type - dict. The parameters of the system. Keys are the parameter names and values are their value.
        traits: Type - dict. Traits of the system. Keys are the trait names and values are their value.
        BHD_ODEs: Type - function. A function that defines the RHS of the BHDs. The function should have inputs (in this order) of state variable, parameters, traits, which should all be vector floats in the order specified in the corresponding dictionaries
        M: Type - int. Number of repeats of the simulation to average over.
        '''

        # Add any inputs to the class
        self.stateVars = stateVars
        self.params = params
        self.traits = traits
        self.BHD_ODEs = BHD_ODEs
        self.M = M
        self.vol = vol

        # Add further variables
        self.popTime = 0
        self.state = np.array(list(stateVars.values()))/self.vol  # The initial state
        self.parVals = list(params.values())  # The parameter values
        self.traitVals = list(traits.values())  # The trait values

        # Storage lists
        self.tStore = [self.popTime]
        self.stateStore = [self.state]
        self.simTime = 0
        self.tBefore = 0
        self.tAfter = 0
        self.recInd = 0
        self.infList = []
        self.recList = []
        self.remList = []
        self.currentInd = 1

    def RHSDynamics(self, state, t):
        '''
        Dynamics for progressing the BHDs
        '''

        return(np.array(self.BHD_ODEs(state, self.parVals, self.traitVals)))
        
    def updateODEs(self, dt):
        '''
        Method to update the dynamics using an implicit Euler regime
        '''

        # Update the state
        # self.state = root(lambda X: X - self.RHSDynamics(X, self.popTime)*dt - self.state, self.state).x
        k1 = dt*self.RHSDynamics(self.state, self.popTime)
        k2 = dt*self.RHSDynamics(self.state + k1/2, self.popTime + dt/2)
        k3 = dt*self.RHSDynamics(self.state + k2/2, self.popTime + dt/2)
        k4 = dt*self.RHSDynamics(self.state + k3, self.popTime + dt)
        self.state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6

        # Update the storage values
        self.popTime += dt
        self.tStore.append(self.popTime)
        self.stateStore.append(self.state)

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

    def plotGraph(self):
        '''
        Plots a directed graph showing the transmission structure
        '''

        # Create a figure
        plt.figure()

        # Create an empty graph
        G = nx.DiGraph()

        # Add all the nodes with the times which initial infection occurred
        G.add_node(0, time=0, color='r', pos=(0,1))
        for ii in range(len(self.infList)):
            G.add_node(self.infList[ii].index, time=self.infList[ii].BHDtime, color='b', pos=(self.infList[ii].index, -self.infList[ii].BHDtime))
        for jj in range(len(self.recList)):
            G.add_node(self.recList[jj].index, time=self.recList[jj].BHDtime, color='g', pos=(self.recList[jj].index, -self.recList[jj].BHDtime))
        for kk in range(len(self.remList)):
            G.add_node(self.remList[kk].index, time=self.remList[kk].BHDtime, color='k', pos=(self.remList[kk].index, -self.remList[kk].BHDtime))

        # Loop through infected individuals and add edges
        for ii in range(len(self.infList)):
            G.add_edge(self.infList[ii].infector, self.infList[ii].index)
        
        # Loop through those that recovered and add edges
        for jj in range(len(self.recList)):
            G.add_edge(self.recList[jj].infector, self.recList[jj].index)

        # Loop through those that recovered and add edges
        for kk in range(len(self.remList)):
            G.add_edge(self.remList[kk].infector, self.remList[kk].index)

        # Extract all positions
        pos = nx.get_node_attributes(G, 'pos')
        col = nx.get_node_attributes(G, 'color')

        # Draw the network
        nx.draw(G, pos, with_labels=True, node_color=col.values())
        plt.show()

        return(G)

    def plotGraphInd(self, ind):
        '''
        This function returns a graph structure which shows the transmission behaviour from an individual.
        It will plot the individual, plus the secondary and tertiary cases.
        '''

        # Create an empty graph
        G = nx.DiGraph()

        # Firstly, find the individual with index specified
        listOfInds = []
        # Look in each of the storage lists
        # We add 1 in each of the below because counting starts at 1
        for jj in range(len(self.infList)):
            if self.infList[jj].index == ind:
                listOfInds.append(self.infList[jj])
        if len(listOfInds) < 1:
            for jj in range(len(self.recList)):
                if self.recList[jj].index == ind:
                    listOfInds.append(self.recList[jj])
        if len(listOfInds) < 1:
            for jj in range(len(self.remList)):
                if self.remList[jj].index == ind:
                    listOfInds.append(self.remList[jj])

        # Extract the indices for those infected by the primary
        secInds = listOfInds[0].infected

        # Loop through each of these and add the secondary cases to listOfInds
        if len(secInds) > 0:
            terIndsTemp = []
            for ii in range(len(secInds)):
                for jj in range(len(self.infList)):
                    if self.infList[jj].index == secInds[ii]:
                        listOfInds.append(self.infList[jj])
                if len(listOfInds) < 1+ii :
                    for jj in range(len(self.recList)):
                        if self.recList[jj].index == secInds[ii]:
                            listOfInds.append(self.recList[jj])
                if len(listOfInds) < 1+ii:
                    for jj in range(len(self.remList)):
                        if self.remList[jj].index == secInds[ii]:
                            listOfInds.append(self.remList[jj])
            
                # Extract the indices for those infected by the secondary
                terIndsTemp.append(listOfInds[-1].infected)

            # Flatten the list of lists containing the tertiary indices
            terInds = [item for sublist in terIndsTemp for item in sublist]

            # Loop through and add each of the tertiary cases to the listOfInds
            if len(terInds) > 0:
                for ii in range(len(terInds)):
                    for jj in range(len(self.infList)):
                        if self.infList[jj].index == terInds[ii]:
                            listOfInds.append(self.infList[jj])
                    if len(listOfInds) < 1+ii+len(secInds) :
                        for jj in range(len(self.recList)):
                            if self.recList[jj].index == terInds[ii]:
                                listOfInds.append(self.recList[jj])
                    if len(listOfInds) < 1+ii+len(secInds):
                        for jj in range(len(self.remList)):
                            if self.remList[jj].index == terInds[ii]:
                                listOfInds.append(self.remList[jj])

        # Add the various nodes to the graph
        for ii in range(len(listOfInds)):
            G.add_node(listOfInds[ii].index, time=listOfInds[ii].BHDtime, color='r', pos=(listOfInds[ii].index, -listOfInds[ii].BHDtime))

        # Loop through individuals and add edges
        for ii in range(1, len(listOfInds)):
            G.add_edge(listOfInds[ii].infector, listOfInds[ii].index)

        # Extract positions and colours
        pos = nx.get_node_attributes(G, 'pos')
        col = nx.get_node_attributes(G, 'color')

        # Draw the network
        # nx.draw(G, pos, with_labels=True, node_color=col.values())
        nx.draw_planar(G, with_labels=True, node_color = col.values())
        plt.show()

if __name__ == '__main__':

    # We will use a density-dependent per-capita birth rate and constant background mortality
    N = 200
    b = 0.5
    d = 0.005
    q = (b-d)/N

    # Initialise a population
    pop = population(
            stateVars = {'S': N/2, 'I': N/2},
            params = {'b': b, 'q': q, 'd': d},
            traits = {},
            BHD_ODEs = lambda S, P, T: [sum(S)*(P[0] - P[1]*sum(S)) - P[2]*S[0], -P[2]*S[1]]
        )
    
    # Run the simulation for a total of 100 days and plot
    for ii in range(10000):
        pop.updateODEs(0.01)
    pop.plot()