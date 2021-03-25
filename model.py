import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

class DuelingDeepQNetwork(keras.Model): # creates a keras.Model class
    def __init__(self, n_actions, fc1_dims, fc2_dims): # required parameters when calling this class

        # we can use super().__init__(), instead too to mean the same function
        super(DuelingDeepQNetwork, self).__init__() # allows for multiple inheritance, so that our agent class can inherit it's properties
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu') #first layer with fc1_dims output
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu') #second layer with fc2_dims output
        self.V = keras.layers.Dense(1, activation=None) #value string, this is the value of the current state
        # for each state or set of states we output 1 value
        self.A = keras.layers.Dense(n_actions, activation=None)#advantage string, tells us what relative advantage each action provides
        # we want to get a value for each action, given our inputs to system
    

    def call(self, state): #so we can use the name of the object to call the functions self.V & so on
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        #perform a transformation, The sum of the Value function - (sum of advantage - average of advantage)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))) # we want to average along the action dimension not batch dimension
        #keepdims is true, the reduced dimensions are retained with length 1.
        #axis = 0 will act on all the rows to each column
        # axis = 1 will act on all the columns in each row
        # our data is [(0, 0, 0)
        #             ,(0, 0, 0)
        #             ,(0, 0, 0)] 
        # so axis = 1 we get average of each distinct action

        return Q

    # may be unnecessary, please experiment with call()
    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
    # put input into Neural Net & compute the advantage of each action
        return A


class ReplayBuffer():  #we have to sample over the agent's set of memories, with a naive implementation where the 
#temporal difference learning directly, it fails to work

    def __init__(self, max_size, input_shape): # constructor
        self.mem_size = max_size    #size of memory
        self.mem_cntr = 0   #memory counter to keep track of the first available memory so we can go back to the beginning
        # and overwritte the oldest memory

        self.state_memory = np.zeros((self.mem_size, *input_shape), # we can use deque instead too
                                        dtype=np.float32)
        # *input_shape == it is expecting a list or tuple of size input_shape, and it will unpack this list or tuple
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        # memory of the done flags received from the environment
        # Value of the terminal state is 0, the agent receives no rewards as the game is over



    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size) #we find out how far into the memory have we filled up, so we get minimum between mem_ctr & mem_size
        batch = np.random.choice(max_mem, batch_size, replace=False) #samples randomly from 0 to max_mem, until batch_size is reached
        #replace = False to make sure we don't sample duplicate memory
        # batch is an array of random numbers

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class Agent():# learning rate for updating the Deep Neural Network, Gamma, is the discount factor, it follows a power law, and 
            # and usually has value of 0.99, so as the further back the memory is the probability of the combinations
            # of states & actions from that memory of happening is lowered
            # Epsilon dictates how often the agent take random actions or greedy actions
            # Early on epsilon is high, but it decays over time but never reaches 0, as we still want the agent to explore
            # The dillema of Explore over exploit
            # Replace is hyperparameter to dictate how often we want to update our target network
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=100000, fname='dueling_dqn.h5', fc1_dims=128,
                 fc2_dims=128, replace=100):
        self.action_space = [0 for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0 #let us know how many times agent has called the learn function, so we know when to update network
        self.memory = ReplayBuffer(mem_size, input_dims) #memory
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims) # Evaluation network
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims) # Target Network
        # If we have 1 same network to evaluate both the actions and the merits of the action, they end up chasing it's own tail
        # Learning becomes highly unstable
        # q_next starts of exactly same as q_eval, but is updated less regularly, in this case once every 100 times

        # We use mean squared error as we want to shift the agent estimates of the action value function to the true ideal value
        # Which is dictated by the reward feedback system
        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done): #store the transitions in our memory, Inheritance from class
        self.memory.store_transition(state, action, reward, new_state, done) # for when calling the function from replay buffer class

    def choose_action(self, observation): # checks whether to use random action or the best estimated action given the state
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation]) # we add another [] to add another dimension to the array, as the DNN only accepts min 2 dimension
            # We can also use np.new_axis command too
            actions = self.q_eval.advantage(state)
            # .advantage is basically getting the best action using the DNN
            action = tf.math.argmax(actions, axis=1).numpy()[0]
            # Chooses the index of the highest value to be chosen as our action

        
        return action

    def learn(self): # if we haven't sample enough memory from batch size we will just go back to main loop
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0: #check if it is time to update the weights of our target network
            self.q_next.set_weights(self.q_eval.get_weights())

        # sample our buffer at the given batch size
        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        # changing q_pred doesn't matter because we are passing states to the train function anyway
        # also, no obvious way to copy tensors in tf2?
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1
