# Hodgkin-Huxley Neuron Model Simulation

A Python implementation of the Hodgkin-Huxley equations to simulate neuron membrane potential dynamics.

## Overview
This code models the electrical activity of a neuron using the classic Hodgkin-Huxley equations. It numerically solves the system of ordinary differential equations describing ion channel dynamics and membrane potential changes using the 4th-order Runge-Kutta (RK4) method.

## Key Features
- Simulation of voltage-gated sodium (Na⁺), potassium (K⁺), and leak channel currents
- Customizable parameters for membrane properties and ion channel conductances
- RK4 integration for stable numerical solutions

## References
- https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
- https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

## Usage
```python
from neuron import Neuron

# Initialize neuron with default parameters
neuron = Neuron()

# Run simulation with parameters:
# dt=0.05 ms, t0=0 ms, tf=50 ms, V0=-50 mV, I0=10 μA/cm²
t, y = neuron.simulate(dt=0.05, t0=0., tf=50., V0=-50., I0=10.)

# Plot membrane potential
plt.plot(t, y[:,0])
plt.show()