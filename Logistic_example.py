# Example code which uses the first test problem from the paper "Efficient coupling of within- and between-host infectious disease dynamics"

#%% Parts for executing the code

from Coupling import transmissionCoupling
from Population import population
from Individual import individual

#%% Packages

import numpy as np
import matplotlib.pyplot as plt

#%% Within-host model set-up

# np.random.seed(145321)
# random.seed(100)

# Number of individuals
N = 200

# We will use logistic growth with intrinsic rate r and carrying capacity K.
r = 5
K = 1e10

# Define the WHDs for the pathogen load
WState = {'P': 1/K}
WParams = {'r': r}
WTraits = {}
WODERHS = lambda S, P, T, t: P[0]*S[0]*(1 - S[0])
WHD_Dynamics = {'WHD_State': WState,
    'WHD_Params': WParams,
    'WHD_Traits': WTraits,
    'WHD_Dynamics': WODERHS}

indiv = individual(
    stateVars = WState,
    params = WParams,
    traits = WTraits,
    WHD_ODEs = WODERHS
)

#%% Host-deomographic dynamics set-up

# We will use a density-dependent per-capita birth rate and constant background mortality
b = 0.5
d = 0.005
q = (b-d)/N

# HDD normalisation
HDDnorm = N

# Define the HDDs
DState = {'S': N-1, 'I': 1, 'R': 0}
DParams = {'b': b, 'q': q, 'd': d}
DTraits = {}
DODERHS = lambda S, P, T: [sum(S)*(P[0]-P[1]*sum(S)*HDDnorm) - P[2]*S[0], -P[2]*S[1], 0]
DStateType = [1, 2, 3]
HDD_Dynamics = {'BHD_State': DState,
    'BHD_Params': DParams,
    'BHD_Traits': DTraits,
    'BHD_Dynamics': DODERHS,
    'BHD_State_Type': DStateType}

#%% Coupling functions

# We only have transmission and virulence. Transmission will be proportional
# to the pathgen load, while the virulence will be proportional to the square
# of the pathogen load
beta1 = 0.1
alpha1 = 0.005
thresh = 1e9
params = {'tran': beta1, 'vir': alpha1, 'threshold': thresh, 'K': K}
traits = {}

# Transmission function
transFun = lambda S, P, T: P[0]*S[0]*P[3]/P[2]

# Virulence function
virFun = lambda S, P, T: P[1]*(S[0]*P[3]/P[2])**2

#%% Other parameters and variables

# Time
tFinal = 100
dt = 0.5
tVec = np.arange(0, tFinal+dt, dt)

# Number of repeats
M = 200


if __name__ == '__main__':

    # Steady state?
    SS = False

    # Create a simulation class
    sim = transmissionCoupling(
        WHD_Dynamics,
        HDD_Dynamics,
        tVec,
        params,
        traits,
        transFun,
        virFun = virFun,
        M = M,
        vol = HDDnorm,
        SS = SS
    )

    # Simulate
    sim.tauSimulation()
    sim.plot()
    plt.show()