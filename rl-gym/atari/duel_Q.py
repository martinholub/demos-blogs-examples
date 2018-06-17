import numpy as np
import random

import keras
from keras.models import load_model, Sequential, Model, model_from_config
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam, RMSprop #Adagrad, RMSProp
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import Callback, CallbackList

from prioritized_memory import PERMemory as Memory
from logger_utils import _L
from collections import deque, Counter
from utils import clipped_error, mean_q

# logger = _L()

class DuelQ(object):
    """This class represents a Double,Dueling,Deep Neural Q-Network.

    It uses architecutres from several Google DeepMind papers:
    https://www.nature.com/articles/nature14236 ... original paper
    https://arxiv.org/pdf/1511.06581.pdf ... dueling DQN
    https://arxiv.org/pdf/1509.06461.pdf ... double DQN
    """
    def __init__(self, learn_rate = 0.001, img_size = (84,84), num_frames = 3,
                action_size = 6, replay_size = 64, max_memory = 20000):
        self.img_size = tuple(img_size) # downsampling image size
        self.num_frames = int(num_frames) # Deterministic frameskip
        self.learn_rate = float(learn_rate) # optimizer learning rate
        self.action_size = int(action_size) # No. of possible actions in env
        self.num_epochs = int(1) # Epoch size used for training
        self.tau = float(0.01) #
        # Memory
        self.replay_size = int(replay_size) # Size of minibatch sample from memory
        self.memory = Memory(int(max_memory)) # deque(maxlen=max_memory)
        # Agent
        self._construct_q_network()

    def _construct_q_network(self):
        """Constructs the desired deep q learning network

        This extends the network architecture found in DeepMind paper. Dueling-DQN
        approach is implemented (see `policy` layer). BatchNormalization and Dropout
        are generaly helpful and were tested. Empirically they did not perfrom well
        (drove Q to very high/low values). I do not know why.

        https://www.nature.com/articles/nature14236 ... paper on DQN
        https://arxiv.org/pdf/1511.06581.pdf ... dueling DQN
        """
        model = Sequential()

        model.add(Convolution2D(filters = 32, kernel_size = (8, 8), strides = (4, 4),
                                input_shape = self.img_size + (self.num_frames, )))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Convolution2D(filters = 64, kernel_size = (4, 4), strides = (2, 2)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Convolution2D(filters = 64, kernel_size = (3, 3), strides = (1, 1)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Flatten())
        flatten = model.layers[-1].output # get output of the Flatten() layer

        # Dueling DQN -- decompose output to Advantage and Value parts
        # V(s): how good it is to be in any given state.
        # A(a): how much better taking a certain action would be compared to the others
        fc1 = Dense(units = 512, activation = None)(flatten)
        advantage = Dense(self.action_size, activation = None)(fc1)
        fc2 = Dense(units = 512, activation = None)(flatten)
        value = Dense(1)(fc2)
        # dueling_type == 'avg'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
        policy = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],
                        output_shape = (self.action_size, ))([advantage, value])

        input_layer = model.input
        self.model = Model(inputs = [input_layer], outputs = [policy])
        # Create identical copy of model, make sure they dont point to same object
        config = self.model.get_config()
        self.target_model = Model.from_config(config)
        self.target_update() # Assure weights are identical.
        del(model)

        losses = [clipped_error] # Use Huber Loss.
        metrics = ["mae", mean_q]
        self.model.compile( loss = losses, optimizer = Adam(lr = self.learn_rate),
                            metrics = metrics)
        self.target_model.compile(  loss = 'MSE', optimizer = Adam(lr = self.learn_rate),
                                    metrics = metrics)

        print(self.model.summary())
        print("Successfully constructed networks.")

    def policy(self, state, epsilon):
        """Selects an optimal action given current model state, unless random

        Epsilon Greedy Policy selects action according to the model unless a random
        action is taken. The value of epsilon is annealed from high (e.g. 1) to low (e.g. 0.1)
        over some given number of steps. This is the exploration/exploitation and represents
        the increase of our certainity about the model's predicitons

        https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
        """
        q_values = self.model.predict( state.reshape(1, *self.img_size, self.num_frames),
                                        batch_size = 1)
        if np.random.rand() < epsilon:
            opt_action = random.choice(range(self.action_size))
        else:
            opt_action = np.argmax(q_values)

        return opt_action, q_values[0, opt_action]

    def predict_batch(self, minibatch, discount):
        """Predict on batch and return target that model will try to fit

        Briefly, we try to make the network (`model`) predict its own output. This can become
        unstable as the sought value is also being changed while we try to
        approach it. To mitigate this, we keep a target network (`target_model`),
        that we update only occasionally.

        Additionaly, we also predict on old states such that we can compute
        `error` that gives a way how to increase importnace of samples that
        our model is the worst at predicting (largest error).

        https://arxiv.org/pdf/1509.06461.pdf
        https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
        """
        # Must work also on batch size 1
        batch_size = len(minibatch)

        # Split batch to parts (s,a,r,s',d)
        # minibatch = np.reshape(minibatch, (batch_size, 5))
        # states, actions, rewards, next_states, dones = np.split(minibatch, 5, axis = 1)
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for _, (state, action, reward, next_state, done) in enumerate(minibatch):
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        ## DOUBLE DQN
        # Use primary network to choose an action
        q_nexts = self.model.predict_on_batch(np.reshape(next_states,
                                            (-1,*self.img_size,self.num_frames)))
        next_actions = np.argmax(q_nexts, axis=1)
        # Use target network to generate q value for that action
        q_targets = self.target_model.predict_on_batch(np.reshape(next_states,
                                            (-1,*self.img_size,self.num_frames)))
        # predict the future discounted reward. target == reward if done
        targets =   rewards + \
                    discount * np.invert(dones).astype(np.float32) * \
                    q_targets[range(batch_size), next_actions]

        # Update only actions for which we have observation
        q_targets[range(batch_size), actions] = targets # update q to future

        # Need this one for error term for memory update
        q_olds = self.model.predict_on_batch(np.reshape(states,
                                            (-1,*self.img_size,self.num_frames)))
        targets_old = q_olds[range(batch_size), actions] # pull out old value of q_hat
        # Get error for updating priorities in the memory
        errors = (abs(targets_old - targets))

        # Reshape as necessary
        states = np.stack(states, axis = 0) # creates new axis
        # q_targets = np.concatenate(q_targets, axis = 0)

        return (states, q_targets, errors)

    def predict_sample(self, state, action, reward, next_state, done, discount):
        """Prediction wrapper for batchsize 1. See predict_batch.
        """
        dummy_batch = [[state, action, reward, next_state, done]]
        state, q_target, error = self.predict_batch(dummy_batch, discount)
        return state, q_target, error

    def replay(self, discount, logger):
        if isinstance(self.memory, (Memory, )):
        ## Prioretized memory style
            minibatch, idxs, is_weights = self.memory.sample(self.replay_size)
            states, q_targets, errors = self.predict_batch(minibatch, discount)
            self.memory.batch_update(idxs, errors)
            #state is x, q_target is y
            loss = self.model.train_on_batch(states, q_targets, sample_weight = is_weights)

            # logger.debug("Sampled Indices: {}".format((idxs)))
            # logger.debug("IS Weights: {}".format((is_weights)))

        else:
            ## Simple Style Draw random minibatch sample from memory
            minibatch=random.sample(self.memory, min(len(self.memory), self.replay_size))
            states, q_targets, errors = self.predict_batch(minibatch, discount)
            #state is x, q_target is y
            loss = self.model.train_on_batch(states, q_targets)

        loss = dict(zip(self.model.metrics_names, loss))
        # logger.info("Loss/Metrics on fit: {}".format(loss))
        return loss["loss"]

    def save_network(self, path):
        # Saves model at specified path as h5 file.
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Succesfully loaded network.")

    def target_update(self):
        """Does hard update of target network wrt to primary network."""
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)

    def target_update_soft(self):
        """Does soft update of target network wrt to primary one

        If used, then this should be called more frequently than hard update
        as only fraction of model weights updated here.
        """
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        target_model_weights = self.tau * model_weights + \
                                (1-self.tau) * target_model_weights
        self.target_model.set_weights(target_model_weights)
