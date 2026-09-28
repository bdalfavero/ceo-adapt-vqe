# Sampling

In this directory, we use the `SampledLinAlgAdapt` class to optimize on hardware.

## XXZ chain

There are three relevant scripts:

1. `xxz_sampling.py` uses only `SampledLinAlgAdapt` to optimize the energy of a Heisenberg XXZ chain.
1. `xxz_handoff.py` starts out doing a few iterations using `LinAlgAdapt`, then the optimization is resumed using
`SampledLinAlgAdapt`. The energies and circuits are stored to a file.
1. `xxz_noisy_simulation.py` takes the circuits generted by running `xxz_handoff.py` and simulates them with exact
and noisy Aer simulators.

`plots.py` visualizes the results.