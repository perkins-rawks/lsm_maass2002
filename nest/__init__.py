import nest
import numpy as np
import pylab

from lsm.nest.utils import get_spike_times
from lsm.utils import windowed_events

# iaf = Integrate and fire
# psc = Postsynaptic currents - The current in a neuron after it passes through a synapse
# A synapse is the space between two neurons which impulses pass by diffusion of
# a neurotransmitter
# exp = exponential

# n_E = Number of excitatory neurons
# n_I = Number of inhibitory neurons

# A refractory period is the time it takes for a neuron
# to rest after responding to a stimulus.

# Integrate and fire neuron model with exponential PSCs
def create_iaf_psc_exp(n_E, n_I):
    # Creates integrated-and-fire post-synaptic exponential current
    # Creates a neuron model (already implemented?) to simulate.
    # The first parameter to Create is the type of neuron you want to simulate.
    # i.e. iaf_psc_exp
    nodes = nest.Create(
        # type of model, size
        'iaf_psc_exp',
        n_E + n_I,  # n_E + n_I is the size 

        # Dictionary of other parameters in a neuron
        {'C_m': 30.0,        # 1.0, Membrane capacity
         'tau_m': 30.0,      # Membrane time constant in ms
         'E_L': 0.0,         # The resting state of neuron(leak reversal potential)
         'V_th': 15.0,       # Spike threshold in mV
         'tau_syn_ex': 3.0,  # Absolute refractory period for excitatory neurons
         'tau_syn_in': 2.0,  # Absolute refractory period for inhibitory neurons                   
         'V_reset': 13.8})   # Reset voltage in mV: the reset value for V_m after a spike

    # SetStatus changes the properties of the nest
    # I_e is the constant external input current
    nest.SetStatus(nodes, [{'I_e': 14.5} for _ in nodes])

    # equivalent / previous code
    # nest.SetStatus(nodes, [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])

    # Return two lists, one of all the inhibitory neurons, and one with all the excitatory
    # ones.
    return nodes[:n_E], nodes[n_E:]

# Tsodyks is the name of a theoretical neuroscientist
# nodes_E is the list of excitatory neurons
# nodes_I is the list of excitatory neurons 


def connect_tsodyks(nodes_E, nodes_I):
    f0 = 10.0

    # Delay of all the connections:
    # mean is 10 
    # std. dev. is 20
    # min is 3
    # max is 200
    delay = dict(distribution='normal_clipped',
                 mu=10., sigma=20., low=3., high=200.)
    # Each neuron is connected to 2 excitatory and 1 inhibitory neurons 
    n_syn_exc = 2  # number of excitatory synapses per neuron
    n_syn_inh = 1  # number of inhibitory synapses per neuron

    # w is the spike-adaptation current in pA (???)
    w_scale = 10.0

    # J? The weight between neurons in juice (brain juice).
    # The larger the magnitude of the weight, the more influential to the output it is.
    # Subscripts are between neurons
    # I to E, I to I, E to E, E to I
    # picoAmperes 10^(-12) ??
    J_EE = w_scale * 5.0    # strength of E->E synapses [pA]
    J_EI = w_scale * 25.0   # strength of E->I synapses [pA]
    J_IE = w_scale * -20.0  # strength of inhibitory synapses [pA]
    J_II = w_scale * -20.0  # strength of inhibitory synapses [pA]

    # U = use
    # Use of neuron (??)

    # D = time constant for depression
    # For both short and longterm plasticity ^
    # Lack of use of neuron causes reduction in the efficacy of neuronal synapses
    # The less active a neuron is, the weaker its synapses are.

    # Postsynaptic Potential (psp) - Synaptic potential from previous neurons,
    # F = time constant for facilitation: the amount of time for neural
    # facilitation to happen (psp is stronger during this time)
    # Neural facilitation: postsynaptic potentials evoked by an impulse are increased
    #                      when that impulse closely follows a prior impulse
    # To do with short term plasticity ^

    # Are these the input function and liquid state? u_0 and t_0

    def get_u_0(U, D, F):
        return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))

    def get_x_0(U, D, F):
        return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))

    # psc? postsynaptic current
    # fac? facilitating current
    # rec? recovery time

    # This function takes in 3 times, and a utilization value, and outputs
    # a dictionary of those values, plus u and x
    def gen_syn_param(tau_psc, tau_fac, tau_rec, U):
        return {"tau_psc": tau_psc,  # postsynaptic current (?) How does time relate?
                "tau_fac": tau_fac,  # facilitation time constant in ms
                "tau_rec": tau_rec,  # recovery time constant in ms (???)
                "U": U,  # utilization
                # u_0: (U, D, F)
                "u": get_u_0(U, tau_rec, tau_fac),
                "x": get_x_0(U, tau_rec, tau_fac),
                }

    # Connect takes:
    # src?        source (source of action potential)
    # trg?        target (action potential receiver)
    # J?          strength between synapses
    # n_syn?      number of synapses
    # syn_param?  the synapse hyper parameters
    def connect(src, trg, J, n_syn, syn_param):
        # from https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_apis.html#module-nest.lib.hl_api_connections
        nest.Connect(src, trg,
                     # connection generator 
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
                     # mu is the mean
                     # sigma is the std. dev.
                     dict({'model': 'tsodyks_synapse', 'delay': delay,
                           'weight': {"distribution": "normal_clipped", "mu": J, "sigma": 0.7 * abs(J),
                                      "low" if J >= 0 else "high": 0.
                                      }},
                          **syn_param))

    # Notice we are now outside of the connect function

    # Is every neuron in the liquid layer connected to every other neuron in the
    # liquid layer?
    connect(nodes_E, nodes_E, J_EE, n_syn_exc, gen_syn_param(
        tau_psc=2.0, tau_fac=1.0, tau_rec=813., U=0.59))
    connect(nodes_E, nodes_I, J_EI, n_syn_exc, gen_syn_param(
        tau_psc=2.0, tau_fac=1790.0, tau_rec=399., U=0.049))
    connect(nodes_I, nodes_E, J_IE, n_syn_inh, gen_syn_param(
        tau_psc=2.0, tau_fac=376.0, tau_rec=45., U=0.016))
    connect(nodes_I, nodes_I, J_II, n_syn_inh, gen_syn_param(
        tau_psc=2.0, tau_fac=21.0, tau_rec=706., U=0.25))


def inject_noise(nodes_E, nodes_I):
    p_rate = 25.0  # this is used to simulate input from neurons around the populations
    J_noise = 1.0  # strength of synapses from noise input [pA]
    delay = dict(distribution='normal_clipped',
                 mu=10., sigma=20., low=3., high=200.)

    noise = nest.Create('poisson_generator', 1, {'rate': p_rate})

    nest.Connect(noise, nodes_E + nodes_I, syn_spec={'model': 'static_synapse',
                                                     'weight': {
                                                         'distribution': 'normal',
                                                         'mu': J_noise,
                                                         'sigma': 0.7 * J_noise
                                                     },
                                                     # made this UNUSED variable
                                                     # delay, just use it here
                                                     'delay': dict(distribution='normal_clipped',
                                                                   mu=10., sigma=20.,
                                                                   low=3., high=200.)
                                                     })

# object means that LSM is at the top of the hierarchy of objects
class LSM(object):
    
    # The parameters are the functions from earlier in the program
    def __init__(self, n_exc, n_inh, n_rec,
                 create=create_iaf_psc_exp, connect=connect_tsodyks, inject_noise=inject_noise):

        # Calls the create function and receives a list of excitetory and
        # inhibitory neuron. 
        neurons_exc, neurons_inh = create(n_exc, n_inh)

        # Connects the excitatory and inhibitory neurons by calling the connect function.
        connect(neurons_exc, neurons_inh)

        # Calls the inject noise function to create noises.
        inject_noise(neurons_exc, neurons_inh)

        # Parameters.
        self.exc_nodes = neurons_exc
        self.inh_nodes = neurons_inh

        # Input nodes = excitetory neurons 
        self.inp_nodes = neurons_exc

        # recorded neurons 
        self.rec_nodes = neurons_exc[:n_rec]

        # number of excitetory, inhibitory, and recorded neurons respectively 
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_rec = n_rec

        # readout nodes == record detector
        self._rec_detector = nest.Create('spike_detector', 1)

        # connects the recorded and the readout neurons
        nest.Connect(self.rec_nodes, self._rec_detector)

    def get_states(self, times, tau):

        # Gets a list of numpy arrays.
        spike_times = get_spike_times(self._rec_detector, self.rec_nodes)

        # 
        return LSM._get_liquid_states(spike_times, times, tau)

    @staticmethod
    # states   -> X
    # targets  -> b
    # reg_fact -> regularization fact, lambda from paper
    def compute_readout_weights(states, targets, reg_fact=0):
        """
        Train readout with linear regression
        :param states: numpy array with states[i, j], the state of neuron j in example i
        :param targets: numpy array with targets[i], while target i corresponds to example i
        :param reg_fact: regularization factor; 0 results in no regularization
        :return: numpy array with weights[j]
        """
        if reg_fact == 0:
            # lstsq solves the equation Xw = b for the best w 
            w = np.linalg.lstsq(states, targets)[0]
        else:
            # pylab.inv -> inverse 
            # pylab.eye -> identity matrix
            # Note that the inverse of kI_n = 1/k I_n for a scalar k. 
            
            # This is somewhat related to the least squares equation.
            # A^TA x = A^T b 
            # for vectors x and b 
            w = np.dot(np.dot(pylab.inv(reg_fact * pylab.eye(np.size(states, 1)) + np.dot(states.T, states)),
                              states.T),
                       targets)
        return w

    @staticmethod
    def compute_prediction(states, readout_weights):
        # Computes b_i (prediction) by multiplying X * w_i
        return np.dot(states, readout_weights)

    @staticmethod
    def _get_liquid_states(spike_times, times, tau, t_window=None):

        # gets the number of neurons.
        # Second parameter is the axis
        # RECORDED NEURONS
        n_neurons = np.size(spike_times, 0)

        # Gets the number of times at which samples are taken?
        n_times = np.size(times, 0)

        # Creates a (n_times) x (n_neurons) matrix -> the X matrix
        states = np.zeros((n_times, n_neurons))
        
        if t_window is None:
            t_window = 3 * tau
            
        # neuron, spike time 
        for n, spt in enumerate(spike_times):
            # To do state order is reversed, as windowed_events are provided in reversed order
            for i, (t, window_spikes) in enumerate(windowed_events(np.array(spt), times, t_window)):
                # This index means we're going backwards 
                # n_times is the row size of X
                # - i -> it's going backwards
                states[n_times - i - 1, 
                       n] = sum(np.exp(-(t - window_spikes) / tau)) # the f function 
        return states
