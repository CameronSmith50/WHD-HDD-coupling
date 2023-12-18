# Within-Between hybrid

Accompanying code for the paper: "Efficient coupling of within- and between-host
infectious disease dynamics" by Cameron A. Smith and Ben Ashby

Maintained by: Cameron Smith, University of Oxford

# Usage

To run a basic simulation uses the transmissionCoupling class within
Coupling.py. A sample file is given in Logistic_example.py. For a more detailed
description of the main classes

## Inidvidual.py

This contains the class for each individual. This is used in order to determine
the current within-host state of an individual, and can also be used to plot
their within-host trajectories and track any genotype labels. The basic syntax
for the required arguments is as follows

```python
from Individual import individual
indiv = individual(
        stateVars = {'P': 1e-10},
        params = {'r': 5},
        traits = {},
        WHD_ODEs = lambda S, P, T, t: S[0]*P[0]*(1-S[0])
    )
```

where the state variables and their initial conditions are defined in the
stateVars dictionary: {variable_name: initial_condition}, parameters are given in
the params dictionary: {param_name: value}, any traits (evolving quantities) are
placed in the traits dictionary and the WHD_ODEs are a function which contains
the state variable (S), parameter values (P) and trait values (T), as well as
explicit time as an input, and outputs the RHS of the ODEs. Other optional
arguments are described in the Individual.py file

## Population.py

This contains the class for a population of individual. Allows us to compute
population-level trajectories of prevalence (for example), as well as infection
networks. The basic syntax for the required arguments is as follows:

```python
from Population import population
pop = population(
        stateVars = {'S': N/2, 'I': N/2},
        params = {'b': b, 'q': q, 'd': d},
        traits = {},
        BHD_ODEs = lambda S, P, T: [sum(S)*(P[0] - P[1]*sum(S)) - P[2]*S[0], -P[2]*S[1]]
    )
```

All of these arguments have the same description as in Individual.py

## Coupling.py

The script Logistic_example.py has an in depth example of how the coupling class works.