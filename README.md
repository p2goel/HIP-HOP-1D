# HIP-HOP-1D

Welcome to HIP-HOP-1D, a miniature program to perform Tully's Fewest-Switches Surface Hopping (FSSH) to simulate nonadiabatic dynamics for one-dimensional model systems. 

Currently, standard decoherence-corrected FSSH is implemented with momentum adjustments, but more features will be added in future.

## What you need to run the program

- python3
- numpy
- scipy
- autograd (pip install autograd)

If your system is running default python2, you can install python3 alongside it and make use of virtual enrionment. This can be easily done by Anaconda distribution, for example. 

## Defining the model potential

The user has to define the model potential in diabatic picture in input_pes.py. Each element of the diabatic potential energy matrix, both diagonal and off-diagonal, need to be defined individually. One can use lambda function for compactness. At the end, one needs to collect all the diabats (diagonal elements) and all the couplings (off-diagonal elements) in two separate lists. This step is important as this defines the vibronic model for the program. For a three state superexchange model, the input looks like the following:

```
V11 = lambda x: 0.
V22 = lambda x: 0.01
V33 = lambda x: 0.005
V12 = lambda x: 0.001*np.exp(-0.5*(x**2))
V23 = lambda x: 0.01*np.exp(-0.5*(x**2))
V31 = lambda x: 0.

all_diabats = [V11, V22, V33]
all_couplings = [V12, V23, V31]
```

## Running a single trajectory

If you just want to run a single trajectory, you can call the molecular_dynamics program as follows:

```
python molecular_dynamics.py -4.0 0.02 1
```

Where the first argument is initial position, second is the initial velocity (not momenta), and third is just an integer (indicating the trajctory number).

The input for SH dynamics must be defined in input_dynamics.py. The keywords are self-explanatory. This run will create three output files:

```
output_sh_dyn_1
md_data_1
populations_1
```

The main output file keeps track of hopping and one can see this by doing ``` grep Hopping output_sh_dyn_1``` (case sensitive). The md_data file contains all the relevant information on dynamics as a function of time. The populations file writes all the electronic state populations.

## Running a swarm of trajectories

Usually one would be interested in running a swarm of (independent) trajectories and perform statistical averaging as that is how SH tries to emulate a true quantum nuclear wavpacket behaviour. The script run_sh.py is provided for this purpose. The user needs to define the number of trajectories, initial position and momentum, and a name for the output directoy where all the output data will be stored. One can then simply run:

```
python run_sh.py
```

## Statistical analysis

An important thing to check in any SH simulation is the so-called "internal consistency", where the average population of an electronic state should match the average occupation (fraction of trajectories in that state) at a given time. This can be accomplished by analysis.py.

## Examples

Two examples are provided in Tests. 

- Tully's simple crossing model (two state)
- Superexchange model (three state)


