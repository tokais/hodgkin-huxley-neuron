import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, C = 1.0, E_Na = 115.0, E_K = -12.0, E_L = 10.6, g_Na = None, g_Na_max = 120.0,
                 g_K = None, g_K_max = 36.0, g_L = None, g_L_max = 0.3):
        """
        Initialize a Hodgkin-Huxley model neuron.

        Parameters:
            C (float, optional): Membrane capacitance in μF/cm². Default is 1.0.
            E_Na (float, optional): Relative sodium (Na) reversal potential in mV. Default is 115.0.
            E_K (float, optional): Relative potassium (K) reversal potential in mV. Default is -12.0.
            E_L (float, optional):  Relative leak reversal potential in mV. Default is 10.6.
            g_Na (callable, optional): Sodium conductance function (mS/cm²) as a function of voltage (V).
                 If None, uses a constant conductance equal to g_Na_max. Default is None.
            g_Na_max (float, optional): Maximum sodium conductance (mS/cm²) used if g_Na is None. Default is 120.0.
            g_K (callable, optional): Potassium conductance function (mS/cm²) as a function of voltage (V).
                If None, uses a constant conductance equal to g_K_max. Default is None.
            g_K_max (float, optional): Maximum potassium conductance (mS/cm²) used if g_K is None. Default is 36.0.
            g_L (callable, optional): Leak conductance function (mS/cm²) as a function of voltage (V).
                If None, uses a constant conductance equal to g_L_max. Default is None.
            g_L_max (float, optional): Maximum leak conductance (mS/cm²) used if g_L is None. Default is 0.3.
        """
        self.C = C
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.g_Na = lambda V: g_Na_max if g_Na is None else g_Na(V)
        self.g_K = lambda V: g_K_max if g_K is None else g_K(V)
        self.g_L = lambda V: g_L_max if g_L is None else g_L(V)
    
    def simulate(self, dt, t0, tf, V0, I0):
        """
        Simulate the neuron's behavior over time using the Hodgkin-Huxley model and RK4 integration.

        Parameters:
            dt (float): Time step size in milliseconds.
            t0 (float): Initial simulation time in milliseconds.
            tf (float): Final simulation time in milliseconds.
            V0 (float): Relative initial membrane potential in mV.
            I0 (callable): Function returning applied current in μA/cm² for given time parameter (ms).

        Returns:
            tuple: A tuple (t, y) where:
                t (numpy.ndarray): Array of time points from t0 to tf with step dt.
                y (numpy.ndarray): Array of shape (n_steps+1, 4) containing the state variables [V, n, m, h] at each time step,
                                   where V is membrane potential (mV), and n, m, h are gating variables.
        """
        n_inf, tau_n = self.inf_and_tau(self.alpha_n, self.beta_n)
        m_inf, tau_m = self.inf_and_tau(self.alpha_m, self.beta_m)
        h_inf, tau_h = self.inf_and_tau(self.alpha_h, self.beta_h)
        
        def hodgkin_huxley(t, y):
            V, n, m, h = y
            dVdt = ((n**4 * self.g_K(V) * (self.E_K - V)) +
                    (m**3 * h * self.g_Na(V) * (self.E_Na - V)) +
                    (self.g_L(V) * (self.E_L - V)) +
                    I0(t)) / self.C
            dndt = (n_inf(V) - n) / tau_n(V)
            dmdt = (m_inf(V) - m) / tau_m(V)
            dhdt = (h_inf(V) - h) / tau_h(V)
            return [dVdt, dndt, dmdt, dhdt]
        
        y0 = [V0, n_inf(V0), m_inf(V0), h_inf(V0)]
        t, y = self._rk4(hodgkin_huxley, t0, y0, dt, int((tf-t0)/dt))
        return t, y
        
    def inf_and_tau(self, alpha, beta):
        x_inf = lambda V: alpha(V) / (alpha(V) + beta(V))
        tau_x = lambda V: 1. / (alpha(V) + beta(V))
        return x_inf, tau_x
    
    def alpha_n(self, V):
        return (0.01*(10-V)) / (np.exp((10-V)/10)-1)
    
    def beta_n(self, V):
        return 0.125 * np.exp(-V/80)
    
    def alpha_m(self, V):
        return (0.1*(25-V)) / (np.exp((25-V)/10)-1)
    
    def beta_m(self, V):
        return 4.0 * np.exp(-V/18)
    
    def alpha_h(self, V):
        return 0.07 * np.exp(-V/20)
    
    def beta_h(self, V):
        return 1. / (np.exp((30-V)/30) + 1)
    
    def _rk4(self, f, t0, y0, dt, n_steps):
        t = t0
        y = np.asarray(y0)
        ts = [t]
        ys = [y.copy()]
        
        for _ in range(n_steps):
            k1 = np.array(f(t, y))
            k2 = np.array(f(t + dt/2, y + (dt * k1)/2))
            k3 = np.array(f(t + dt/2, y + (dt * k2)/2))
            k4 = np.array(f(t + dt, y + dt * k3))

            y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            t += dt
            
            ts.append(t)
            ys.append(y.copy())
        
        return np.array(ts), np.array(ys)


def main():
    neuron = Neuron()
    t, y = neuron.simulate(.05, 0., 650., 0., lambda t: 0. if (t%200) < 100 else 40.)
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    ax1.plot(t, y[:, 0])
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Hodgkin-Huxley Neuron Dynamics')
    ax1.grid(True)
    ax2.plot(t, y[:, 1], label='n')
    ax2.plot(t, y[:, 2], label='m')
    ax2.plot(t, y[:, 3], label='h')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Gating Variable Value')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
