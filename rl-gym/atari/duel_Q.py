import numpy as np
import random

import keras
from keras.models import load_model, Sequential, Model, model_from_config
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam, RMSprop #Adagrad, RMSProp
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import Input, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import TensorBoard

from prioritized_memory import PERMemory as Memory
from logger_utils import _L
from collections import deque, Counter
from utils import clipped_masked_error, mean_q, clipped_error

# logger = _L()

class DuelQ(object):
    """This class represents a Double,Dueling,Deep Neural Q-Network.

    It uses architecutres from several Google DeepMind papers:
    https://www.nature.com/articles/nature14236 ... original paper
    https://arxiv.org/pdf/1511.06581.pdf ... dueling DQN
    https://arxiv.org/pdf/1509.06461.pdf ... double DQN
    """
    def __init__(self, learn_rate = 0.001, img_size = (84,84), num_frames = 3,
                action_size = 6, replay_size = 64, max_memory = 20000, is_test = False,
                num_episodes = 10):
        self.img_size = tuple(img_size) # downsampling image size
        self.num_frames = int(num_frames) # Deterministic frameskip
        self.learn_rate = float(learn_rate) # optimizer learning rate
        self.action_size = int(action_size) # No. of possible actions in env
        self.num_epochs = int(1) # Epoch size used for training
        self.tau = float(0.01) #
        self.is_test = bool(is_test)
        # Memory
        self.replay_size = int(replay_size) # Size of minibatch sample from memory
        self.memory = Memory(int(max_memory)) # deque(maxlen=max_memory)
        # self.memory = deque(maxlen=max_memory)
        self.num_episodes = int(num_episodes)
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

        Notes:
          Using Batch Normlization requires same size of batch on training and
          test. This cannot be easily implemented in RL scenario with PER.
        """
        # Sequential model.add replaced with x = (...)(x) functional API
        # model = Sequential()

        #  Mask that allows updating of only action that was observed
        mask_input = Input((self.action_size, ), name = 'mask')
        #  Preprocess data on input, allows storing as uint8
        frames_input = Input(self.img_size + (self.num_frames, ), name = 'frames')
        # Scale by 142 instead of 255, because for BreakOut the max val is 142
        x = (Lambda(lambda x: x / 142.0)(frames_input))

        # DEBUG: All filters, and units halved to make easier to train

        x = (Convolution2D(filters = 16, kernel_size = (8, 8), strides = (4, 4),
                                # input_shape = self.img_size + (self.num_frames, ),
                                kernel_regularizer = l2(0.1),
                                kernel_initializer = 'he_normal'))(x)
        # model.add(BatchNormalization())
        x = (Activation('relu'))(x)
        # if not is_test: model.add(Dropout(0.5))

        x = (Convolution2D(filters = 32, kernel_size = (4, 4), strides = (2, 2),
                                kernel_regularizer = l2(0.1),
                                kernel_initializer = 'he_normal'))(x)
        # model.add(BatchNormalization())
        x = (Activation('relu'))(x)
        # if not is_test: model.add(Dropout(0.5))

        # DEBUG: Removed Third layer to make model smaller
        # x = (Convolution2D(filters = 32, kernel_size = (3, 3), strides = (1, 1),
        #                         kernel_regularizer = l2(0.01)))(x)
        # # model.add(BatchNormalization())
        # x = (Activation('relu'))(x)
        # if not is_test: model.add(Dropout(0.5))

        flatten = (Flatten())(x)
        # flatten = model.layers[-1].output # get output of the Flatten() layer

        # Dueling DQN -- decompose output to Advantage and Value parts
        # V(s): how good it is to be in any given state.
        # A(a): how much better taking a certain action would be compared to the others
        fc1 = Dense(units = 128, activation = None, kernel_regularizer = l2(0.1),
                    kernel_initializer = 'he_normal')(flatten)
        advantage=Dense(self.action_size, activation = None,
                        kernel_regularizer = l2(0.1),kernel_initializer = 'he_normal')(fc1)
        # DEBUG: Simplify the net
        # fc2 = Dense(units = 512, activation = None, kernel_regularizer = l2(0.01))(flatten)
        # value = Dense(1, kernel_regularizer = l2(0.01))(fc2)
        # dueling_type == 'avg'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
        # policy = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],
        #                 output_shape = (self.action_size, ))([advantage, value])
        policy = advantage
        filtered_policy = multiply([policy, mask_input])

        self.model = Model(inputs = [frames_input, mask_input], outputs = [filtered_policy])
        # Create identical copy of model, make sure they dont point to same object
        config = self.model.get_config()
        self.target_model = Model.from_config(config)
        self.target_update() # Assure weights are identical.

        # losses = [clipped_masked_error(mask_input)] # Use Huber Loss.
        losses = ["MSE"] # DEBUG
        metrics = ["mae", mean_q]

        # optimizer = Adam(   lr = self.learn_rate,
        #                     epsilon = 0.01,
        #                     decay = 1e-5,
        #                     clipnorm = 1.)
        optimizer = RMSprop(lr = self.learn_rate,
                            epsilon = 0.00,
                            rho = 0.99,
                            decay = 1e-6,
                            clipnorm = 1.)
        self.model.compile( loss = losses,
                            optimizer = optimizer,
                            metrics = metrics)
        #  Loss, optimizer and metrics just dummy as never trained
        self.target_model.compile(  loss = 'MSE',
                                    optimizer = Adam(),
                                    metrics = [])

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
        all_one_mask = np.ones((1, self.action_size), dtype = np.uint8)
        q_values=self.model.predict([state.reshape(1, *self.img_size, self.num_frames),
                                    all_one_mask], batch_size = 1)
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
        all_one_mask = np.ones((len(actions), ) + (self.action_size, ),
                                dtype = np.uint8)
        q_nexts = self.model.predict_on_batch([np.reshape(next_states,
                                            (-1,*self.img_size,self.num_frames)),
                                            all_one_mask])
        next_actions = np.argmax(q_nexts, axis=1)
        # Use target network to generate q value for that action
        q_targets = self.target_model.predict_on_batch([np.reshape(next_states,
                                            (-1,*self.img_size,self.num_frames)),
                                            all_one_mask])
        # predict the future discounted reward. target == reward if done
        targets =   rewards + \
                    discount * np.invert(dones).astype(np.float32) * \
                    q_targets[range(batch_size), next_actions]

        # Update only actions for which we have observation
        # This is simultanously implemented on model level. Should not be an issue
        # In future remove from here.
        q_targets[range(batch_size), actions] = targets # update q to future

        # Need this one for error term for memory update
        q_olds = self.model.predict_on_batch([np.reshape(states,
                                            (-1,*self.img_size,self.num_frames)),
                                            all_one_mask])
        targets_old = q_olds[range(batch_size), actions] # pull out old value of q_hat
        # Get error for updating priorities in the memory
        errors = (abs(targets_old - targets))

        #Get mask to update only q_value that was observed:
        mask = np.zeros((batch_size, self.action_size), dtype = np.uint8)
        mask[range(batch_size), actions] = 1

        # Reshape as necessary
        states = np.stack(states, axis = 0) # creates new axis
        # q_targets = np.concatenate(q_targets, axis = 0)

        return (states, q_targets, errors, mask)

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
            states, q_targets, errors, mask = self.predict_batch(minibatch, discount)
            self.memory.batch_update(idxs, errors)
            #state is x, q_target is y
            # tr_id = np.arange(0, self.replay_size*(1-0.2))
            # vl_id = np.arange(self.replay_size*0.2, self.replay_size, dtype= np.uint8)
            history = self.model.fit([states, mask], q_targets,
                                    batch_size = len(minibatch),
                                    validation_split = 0.2, verbose = 0)
                                    # validation_data = ([states[vl_id,:], mask[vl_id,:]], q_targets[vl_id,:]),


            # logger.debug("Sampled Indices: {}".format((idxs)))
            # logger.debug("IS Weights: {}".format((is_weights)))

        else:
            ## Simple Style Draw random minibatch sample from memory
            minibatch=random.sample(self.memory, min(len(self.memory), self.replay_size))
            states, q_targets, errors, mask = self.predict_batch(minibatch, discount)
            #state is x, q_target is y
            # history = self.model.train_on_batch([states, mask], q_targets)
            # mock validation to enable TensorBoard callback visualizations
            history = self.model.fit([states, mask], q_targets, batch_size = len(minibatch),
                                    validation_split = 0.2, verbose=0)

            # TODO: merge the validation and training dictionary for later consistency

        # history = dict(zip(self.model.metrics_names, loss))
        # logger.info("Loss/Metrics on fit: {}".format(loss))
        metrics = []
        for met in self.model.metrics_names:
            metrics.extend(history.history[met])
        return np.array(metrics), history.validation_data

    def save_network(self, path):
        # Saves model at specified path as h5 file.
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        try:
            custom = {"clipped_error":clipped_error, "mean_q":mean_q}
            model = load_model(path, custom_objects=custom)
            self.model = model
            self.target_update()
        except ValueError as e:
            print("Path dos not contain full model, loading weights only.")
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
