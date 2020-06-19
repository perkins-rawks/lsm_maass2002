import nest             # Library for creating a network of biologically inspired neurons
import numpy as np      # Library for matrix multiplication / vectorized functions
import numpy.ma as ma   # Library for masked array

# File reading imports
from lsm.nest import LSM  
from lsm.utils import poisson_generator

# convention: all times in [ms], except stated otherwise


def generate_stimulus_xor(stim_times, gen_burst, n_inputs=2):

    # Input states is a n_inputs (2) x (stim_times element count) matrix
    # Inputs are only 0s and 1s because 2 is the upper bound.
    inp_states = np.random.randint(2, size=(n_inputs, np.size(stim_times)))

    # Input spikes is 
    inp_spikes = []

    # ma - invalid options inside Arrays
    # Everything in inp_states that is close to 0 is invalid (canceling
    # insignificant spikes?)
    # What does this matrix multiplication accomplish? 
    # -> A way of randomizing the times, only multiply 1 * values in stim_times. 
    for times in ma.masked_values(inp_states, 0) * stim_times:
        # for each input (neuron): generate spikes according to state (=1) and
        # stimulus time-grid
        # gen_burst() is a parameter function that gets passed the Poisson generator
        spikes = np.concatenate([t + gen_burst() for t in times.compressed()])

        # round to simulation precision
        spikes *= 10
        spikes = spikes.round() + 1.0
        spikes = spikes / 10.0

        inp_spikes.append(spikes)

    # astype(int) could be omitted, because False/True has the same semantics
    # *inp_states breaks the matrix down into the 2 rows
    # Computes row 1 XOR row 2 elementwise
    targets = np.logical_xor(*inp_states).astype(int)

    # inp_spikes is a list of spikes, and target is an int
    return inp_spikes, targets

# neuron_targets are the input layer
def inject_spikes(inp_spikes, neuron_targets):

    # Creates a spike generator device for as long of time as there are spikes 
    spike_generators = nest.Create("spike_generator", len(inp_spikes))

    for sg, sp in zip(spike_generators, inp_spikes):
        nest.SetStatus([sg], {'spike_times': sp})

    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)
    C_inp = 100  # int(N_E / 20)  # number of outgoing input synapses per input neuron

    nest.Connect(spike_generators, neuron_targets,
                 {'rule': 'fixed_outdegree',
                  'outdegree': C_inp}, # each input neuron is connected to 100 others
                 {'model': 'static_synapse',
                  'delay': delay, # normal clipped distr delay
                  'weight': {'distribution': 'uniform', # random weights according to this uniform dist
                             'low': 2.5 * 10 * 5.0,
                             'high': 7.5 * 10 * 5.0}
                  })


def main():
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 11})

    sim_time = 200000

    # stimulus
    # A time step
    # Start at 300 ms, and go to the end of the video at steps of 300
    # 300, 600, 900, ...
    stim_interval = 300

    # How often we take a sample
    stim_length = 50 

    # How far ahead in the video we try to predict ??
    # rate?

    # Frequency in hertz
    stim_rate = 200  # [1/s]

    # delta sample ??
    readout_delay = 10

    # Returns evenly spaced value within an interval
    # Picking stimulus times within the interval   
    # start, stop, step
    stim_times = np.arange(stim_interval, sim_time - stim_length - readout_delay, stim_interval)
    readout_times = stim_times + stim_length + readout_delay

    def gen_stimulus_pattern(): return poisson_generator(stim_rate, t_stop=stim_length)

    # input spikes is a list of spikes and targets is an int (either 0 or 1)
    inp_spikes, targets = generate_stimulus_xor(stim_times, gen_burst=gen_stimulus_pattern)

    # Excitatory = 1000, inhibitory = 250, recorded = 500
    lsm = LSM(n_exc=1000, n_inh=250, n_rec=500)

    # random spikes are injected to the input neurons where each are connected
    # to 100 liquid(excitetory) neurons.
    inject_spikes(inp_spikes, lsm.inp_nodes)

    # SIMULATE
    nest.Simulate(sim_time)

    readout_times = readout_times[5:]
    targets = targets[5:]

    states = lsm.get_states(readout_times, tau=20)

    # add constant component to states for bias (1) (TODO why?)
    # hstack -> concatenates 2 matrices into one horizontal vector
    states = np.hstack([states, np.ones((np.size(states, 0), 1))])

    n_examples = np.size(targets, 0)
    n_examples_train = int(n_examples * 0.8)

    # split the states into the training states and the testing
    # 80% training, 20% testing
    train_states, test_states = states[:n_examples_train, :], states[n_examples_train:, :]
    train_targets, test_targets = targets[:n_examples_train], targets[n_examples_train:]

    readout_weights = lsm.compute_readout_weights(train_states, train_targets, reg_fact=5.0)

    def classify(prediction):
        return (prediction >= 0.5).astype(int)

    # Training 

    # X * w 
    train_prediction = lsm.compute_prediction(train_states, readout_weights)

    # classifiesthe prediction as 0s and 1s
    train_results = classify(train_prediction)


    # Testing 
    test_prediction = lsm.compute_prediction(test_states, readout_weights)
    test_results = classify(test_prediction)

    print("simulation time: {}ms".format(sim_time))
    print("number of stimuli: {}".format(len(stim_times)))
    print("size of each state: {}".format(np.size(states, 1)))

    print("---------------------------------------")

    def eval_prediction(prediction, targets, label):
        # n_successes = sum(prediction == targets) ?
        n_fails = sum(abs(prediction - targets))
        n_total = len(targets)
        print("mismatched {} examples: {:d}/{:d} [{:.1f}%]".format(label, n_fails, n_total, n_fails / n_total * 100))

    eval_prediction(train_results, train_targets, "training")
    eval_prediction(test_results, test_targets, "test")


if __name__ == "__main__":
    main()
