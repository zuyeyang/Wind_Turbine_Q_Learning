# Wind_Turbine_Q_Learning
Research and development of structural control devices have gained great attention in recent years as a method to protect structures against natural hazards. Efforts by structural engineers in an attempt to quantify structural dynamics and viscous damping have achieved success by enacting proxies such as transmissibility, equivalent viscous damping, and damping ratios to deduce maximum forces transferrable to the foundation of a structure. Although existing methods are sufficient to aid in the design of structures via the derivation of steady-state solutions, they are insufficient to capture the cyclical fatigue stemming from the total response of induced vibrations. This study aims to develop an autonomous control framework to attenuate such vibrations applicable to a simplified wind turbine model. Q-learning is used to deduce optimal voltage settings for a semi-active magnetorheological damper under uncertain vorticity conditions. Establishing the optimal voltage setting for the damper, and thus damping ratio, under uncertainty may then provide a trivial target for control systems to tune towards.
## Simulation
`semiactive_damper_simulation.ipynb`
## Data Preprocess
1. Put the data (csv) under the Data => Raw file => assume it named 'data_all.csv'
2. Open the file `Data.ipynb`, import the data at the 2nd cell, change the number of round at this cell
3. run the code all the way done
4. new data input and dictionary will be found at  Data => version_from_python
## Q-learning
Method1: `Final.ipynb` -> choose the floor state for uncertainty state
Method1:`Final.jl` -> choose the nearest neighbour for the uncertainty state
