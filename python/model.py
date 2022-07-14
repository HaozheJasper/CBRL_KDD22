
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from runner_utils import sync_weights
from dopamine.replay_memory.circular_replay_buffer import ReplayElement, OutOfGraphReplayBuffer
import numpy as np
import math
from bert_layer import BertLayer
import copy
LOGSTD_MIN = -20
LOGSTD_MAX = 2
EPS = 1e-6

class NaiveBuffer():
    def __init__(self,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 horizon_length,
                 update_horizon=1, # abandon
                 sync=True,
                 require_nobs=True,
                 use_history=False,
                 history_type='sar',
                 history_size=6,
                 episode_batchsize=5,
                 gamma=0.99,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        # from OutOfGraphReplayBuffer at circular_replay_buffer.py
        assert isinstance(observation_shape, tuple)
        if replay_capacity < update_horizon + stack_size:
            raise ValueError('There is not enough capacity to cover '
                             'update_horizon and stack_size.')

        # logging.info(
        #     'Creating a %s replay memory with the following parameters:',
        #     self.__class__.__name__)
        # logging.info('\t observation_shape: %s', str(observation_shape))
        # logging.info('\t observation_dtype: %s', str(observation_dtype))
        # logging.info('\t terminal_dtype: %s', str(terminal_dtype))
        # logging.info('\t stack_size: %d', stack_size)
        # logging.info('\t replay_capacity: %d', replay_capacity)
        # logging.info('\t batch_size: %d', batch_size)
        # logging.info('\t update_horizon: %d', update_horizon)
        # logging.info('\t gamma: %f', gamma)

        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._state_shape = self._observation_shape # + (self._stack_size,)
        self._replay_capacity = replay_capacity
        self._sync = sync # synchronous or not
        self._require_nobs = require_nobs
        self.use_history = use_history
        self.history_type = history_type
        self.history_size = history_size
        self.sar_size = observation_shape[0]+2 if history_type == 'sar' else 2*observation_shape[0]+2
        self._episode_batchsize = episode_batchsize
        self._batch_size = batch_size
        self._horizon_length = horizon_length
        self._num_episode = (replay_capacity-1)//horizon_length + 1
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._create_storage()
        self.add_count = np.array(0)
        self.episode_cursor = np.array(0) # number of episodes experienced
        self.inep_cursor = np.array(0)
        self.invalid_range = np.zeros((self._stack_size))
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)],
            dtype=np.float32)
        self._next_experience_is_episode_start = True
        self._episode_end_indices = set()

        # storage: field -> ntraj, trajlength
        # cache: several episodes being updated

    # def _create_episode_storage(self):
    #     store = {}
    #     for storage_element in self.get_storage_signature():
    #         array_shape = [self._horizon_length] + list(storage_element.shape)
    #         self._store[storage_element.name] = np.empty(
    #             array_shape, dtype=storage_element.type)
    #     return store

    def _create_storage(self):
        """Creates the numpy arrays used to store transitions.
        """
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._num_episode*self._horizon_length] + list(storage_element.shape)
            self._store[storage_element.name] = np.empty(
                array_shape, dtype=storage_element.type)

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          list of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('terminal', (), self._terminal_dtype)
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements

    def add(self, obs, action, reward, done, *args, **kwargs):
        """
        kwargs:
            epid_in_batch: episode id in current batch of episodes. used in async mode.
            cursor: the current time step.
        """
        self._check_add_types(obs, action.item(), reward, done, *args)
        self._add(obs, action, reward, done, *args, **kwargs)

    def _add(self, *args, epid_in_batch=None, cursor=None):
        """Internal add method to add to the storage arrays.

        Args:
          *args: All the elements in a transition.
        """
        # self._check_args_length(*args)
        transition = {e.name: args[idx]
                      for idx, e in enumerate(self.get_storage_signature())}
        self._add_transition(transition, epid_in_batch=epid_in_batch, cursor=cursor)

    def _add_transition(self, transition, epid_in_batch=None, cursor=None):
        """Internal add method to add transition dictionary to storage arrays.

        Args:
          transition: The dictionary of names and values of the transition
                      to add to the storage.
          epid_in_batch: episode id in current batch of episodes. used in async mode.
          cursor: the current time step.
        """
        # cursor = self.cursor()
        episode_cursor = self._get_episode_cursor(epid_in_batch)
        cursor = self._get_cursor(cursor)
        for arg_name in transition:
            self._store[arg_name][episode_cursor*self._horizon_length+cursor] = transition[arg_name]

        self.add_count += 1
        is_terminal = transition['terminal']
        if self._sync:
            if is_terminal: self.episode_cursor += 1
            self.inep_cursor = (self.inep_cursor +1) % self._horizon_length
        else:
            # assert epid_in_batch is not None
            # assert cursor is not None
            if is_terminal and epid_in_batch==self._episode_batchsize-1: self.episode_cursor += self._episode_batchsize
            if epid_in_batch==0:
                self.inep_cursor = cursor
        # self.invalid_range = invalid_range(
        #     self.cursor(), self._replay_capacity, self._stack_size,
        #     self._update_horizon)
    def _get_episode_cursor(self, epid_in_batch=None):
        if self._sync: # one by one
            return self.episode_cursor % self._num_episode
        else:
            # assert epid_in_batch is not None, 'epid_in_batch is required when async'
            return (self.episode_cursor+epid_in_batch) % self._num_episode

    def _get_cursor(self, cursor=None):
        if self._sync:
            full_idx_in_buffer = self.add_count % (self._num_episode*self._horizon_length) # idx of current buffer, dropped transitions are removed
            return full_idx_in_buffer % self._horizon_length
        else:
            # assert cursor is not None
            return cursor%self._horizon_length

    #######################
    def _check_add_types(self, *args):
        """Checks if args passed to the add method match those of the storage.

        Args:
          *args: Args whose types need to be validated.

        Raises:
          ValueError: If args have wrong shape or dtype.
        """
        self._check_args_length(*args)
        for arg_element, store_element in zip(args, self.get_storage_signature()):
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO(b/80536437). This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                raise ValueError('arg has shape {}, expected {}'.format(
                    arg_shape, store_element_shape))

    def _check_args_length(self, *args):
        """Check if args passed to the add method have the same length as storage.

        Args:
          *args: Args for elements used in storage.

        Raises:
          ValueError: If args have wrong length.
        """
        if len(args) != len(self.get_storage_signature()):
            raise ValueError('Add expects {} elements, received {}'.format(
                len(self.get_storage_signature()), len(args)))

    def ok_to_sample(self):
        tmp = (self._batch_size - 1) // self._horizon_length + 1
        return self.is_full() or self.episode_cursor>= tmp

    #######################
    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        # assert len(indices) == batch_size

        transition_elements = self.get_transition_elements(batch_size)
        # batch_arrays = self._create_batch_arrays(batch_size)
        # for batch_idx, buffer_idx in enumerate(indices):
        batch_arrays = []
        #     ############################
        #     # trajectory_indices = [(buffer_idx + j) % self._replay_capacity
        #     #                       for j in range(self._update_horizon)]
        #     # trajectory_terminals = self._store['terminal'][trajectory_indices]
        #     # is_terminal_transition = trajectory_terminals.any()
        #     # if not is_terminal_transition:
        #     #     trajectory_length = self._update_horizon
        #     # else:
        #     #     # np.argmax of a bool array returns the index of the first True.
        #     #     trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
        #     #                                   0) + 1
        #     # next_state_index = buffer_idx + trajectory_length
        #     # trajectory_discount_vector = (
        #     #     self._cumulative_discount_vector[:trajectory_length])
        #     # trajectory_rewards = self.get_range(self._store['reward'], buffer_idx,
        #     #                                     next_state_index)
        #     ################################
        #     # Fill the contents of each array in the sampled batch.
        #     epid, stepid = buffer_idx//self._horizon_length, buffer_idx%self._horizon_length
        #     assert len(transition_elements) == len(batch_arrays)
        #     for element_array, element in zip(batch_arrays, transition_elements):
        #         if element.name == 'indices':
        #             element_array[batch_idx] = buffer_idx
        #         if element.name == 'next_obs':
        #             element_array[batch_idx] = self._store['observation'][epid, (stepid+1)%self._horizon_length] # nobs of terminal will never be used
        #         elif element.name in self._store.keys():
        #             element_array[batch_idx] = self._store[element.name][epid, stepid]
        #
        epids = indices // self._horizon_length
        if self.use_history:
            required_num = self.history_size
            past_length = (indices) % self._horizon_length
            num_history = [min(l, required_num) for l in past_length]
        else: num_history = None
        for element in transition_elements:
            if element.name == 'indices':
                batch_arrays.append(indices)
            elif element.name == 'next_obs':
                stepids = (indices+1) % self._horizon_length
                batch_arrays.append(self._store['observation'][epids*self._horizon_length+stepids])
            elif element.name == 'history': # batchsize, historysize*sarsize, zero padded
                init = np.zeros((self._batch_size, self.history_size*self.sar_size), dtype=np.float32)
                ind_to_fetch = []
                for idx,num in enumerate(num_history):
                    if num==0: continue
                    epid = epids[idx]
                    stepid = indices[idx] % self._horizon_length
                    ind_to_fetch.extend(list(range(epid*self._horizon_length+(stepid-num),epid*self._horizon_length+(stepid)))[::-1])
                states = self._store['observation'][ind_to_fetch]
                actions = self._store['action'][ind_to_fetch].reshape(-1,1)
                rewards = self._store['reward'][ind_to_fetch].reshape(-1,1)
                concat = np.concatenate((states, actions, rewards), axis=-1) # num, sarsize
                cnt = 0
                for idx,num in enumerate(num_history):
                    if num==0: continue
                    init[idx, :num * self.sar_size] = concat[cnt:cnt+num].reshape(-1)
                    cnt += num
                # for idx,num in enumerate(num_history):
                #     if num==0: continue
                #     epid = epids[idx]
                #     stepid = indices[idx] % self._horizon_length
                #     # store in reversed order, convenient to make next history
                #     states = self._store['observation'][epid*self._horizon_length+(stepid-num):epid*self._horizon_length+(stepid)]
                #     actions = self._store['action'][epid*self._horizon_length+(stepid-num):epid*self._horizon_length+(stepid)].reshape(-1,1)
                #     rewards = self._store['reward'][epid*self._horizon_length+(stepid-num):epid*self._horizon_length+(stepid)].reshape(-1,1)
                #     concat = np.concatenate((states, actions, rewards), axis=-1)[::-1].reshape(-1)  # reverse here
                #     init[idx, :num * self.sar_size] = concat

                batch_arrays.append(init)

            elif element.name == 'num_history':
                batch_arrays.append(np.asarray(num_history))
            elif element.name in self._store:
                epids = indices // self._horizon_length
                stepids = (indices ) % self._horizon_length
                batch_arrays.append(self._store[element.name][epids*self._horizon_length+stepids])
        return tuple(batch_arrays)

    def sample_episode_batch(self, batch_size=16):
        # batch_arrays = []
        epids = self.sample_episode_index_batch(batch_size)
        indices = [(eid+1)*self._horizon_length-1 for eid in epids] # has made sure the sampled episodes are full

        roi_diff = self._store['observation'][indices][:,2] # roi-L
        rbr = self._store['observation'][indices][:-1]
        return roi_diff, rbr

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        max_index = self._num_episode*self._horizon_length
        if self.is_full(): # all columns has been filled
            # remove the columns currently being updated
            if self._sync:
                candidates = [col*self._horizon_length+row for col in range(self._num_episode) for row in range(self._horizon_length) if col!=self.episode_cursor]
            else:
                candidates = [col * self._horizon_length + row for col in range(self._num_episode) for row in
                              range(self._horizon_length) if col not in range(self.episode_cursor, self.episode_cursor+self._episode_batchsize)]

        else:
            candidates = [col * self._horizon_length + row for col in range(self.episode_cursor) for row in
                              range(self._horizon_length)]

        # assert len(candidates)>=batch_size
        indices = np.random.choice(candidates, size=batch_size, replace=False)

        return indices

    def sample_episode_index_batch(self, batch_size=64):
        if self.is_full():
            if self._sync:
                candidates = [col for col in range(self._num_episode) if col!=self.episode_cursor]
            else: raise Exception('not implemented')
        else:
            candidates = list(range(self.episode_cursor))
        indices = np.random.choice(candidates, size=batch_size, replace=len(candidates)<batch_size)
        return indices

    def num_episode_iter(self, batch_size=16):
        if self.is_full(): return self._num_episode//batch_size * 2
        else: return max(self.episode_cursor//batch_size, 1)

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self.add_count >= self._horizon_length*self._num_episode

    def _create_batch_arrays(self, batch_size):
        """Create a tuple of arrays with the type of get_transition_elements.

        When using the WrappedReplayBuffer with staging enabled it is important to
        create new arrays every sample because StaginArea keeps a pointer to the
        returned arrays.

        Args:
          batch_size: (int) number of transitions returned. If None the default
            batch_size will be used.

        Returns:
          Tuple of np.arrays with the shape and type of get_transition_elements.
        """
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = []
        for element in transition_elements:
            batch_arrays.append(np.empty(element.shape, dtype=element.type))
        return tuple(batch_arrays)

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [ # follow storage signature
            ReplayElement('observation', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('action', (batch_size,) + self._action_shape,
                          self._action_dtype),
            ReplayElement('reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            # ReplayElement('next_state', (batch_size,) + self._state_shape,
            #               self._observation_dtype),
            # ReplayElement('next_action', (batch_size,) + self._action_shape,
            #               self._action_dtype),
            # ReplayElement('next_reward', (batch_size,) + self._reward_shape,
            #               self._reward_dtype),
            ReplayElement('terminal', (batch_size,), self._terminal_dtype),
            ReplayElement('indices', (batch_size,), np.int32),
            # ReplayElement('sampling_probabilities', (batch_size,), np.float32)
        ]
        if self._require_nobs:
            transition_elements.append(
                ReplayElement('next_obs', (batch_size,) + self._state_shape,
                          self._observation_dtype))
        if self.use_history:
            transition_elements.extend([ReplayElement('history', (self.sar_size*self.history_size, ), np.float32),
                                  ReplayElement('num_history', (), np.int32)])
        for element in self._extra_storage_types:
            transition_elements.append(
                ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                              element.type))
        return transition_elements

# deprecated
class MyBuffer(OutOfGraphReplayBuffer):
    def __init__(self,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 gamma=0.99,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        super(MyBuffer, self).__init__(observation_shape,
                                       stack_size,
                                       replay_capacity,
                                       batch_size,
                                       update_horizon,
                                       gamma,
                                       max_sample_attempts,
                                       extra_storage_types,
                                       observation_dtype,
                                       terminal_dtype,
                                       action_shape,
                                       action_dtype,
                                       reward_shape,
                                       reward_dtype)

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement('observation', (batch_size,) + self._state_shape, # modify name to avoid shape mismatch
                          self._observation_dtype),
            ReplayElement('action', (batch_size,) + self._action_shape,
                          self._action_dtype),
            ReplayElement('reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('next_obs', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            # ReplayElement('next_action', (batch_size,) + self._action_shape,
            #               self._action_dtype),
            # ReplayElement('next_reward', (batch_size,) + self._reward_shape,
            #               self._reward_dtype),
            ReplayElement('terminal', (batch_size,), self._terminal_dtype),
            ReplayElement('indices', (batch_size,), np.int32),
            # ReplayElement('sampling_probabilities', (batch_size,), np.float32)
        ]
        for element in self._extra_storage_types:
            transition_elements.append(
                ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                              element.type))
        return transition_elements

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        assert len(indices) == batch_size

        transition_elements = self.get_transition_elements(batch_size)
        # batch_arrays = self._create_batch_arrays(batch_size)
        batch_arrays = [[] for _ in transition_elements]
        for batch_element, state_index in enumerate(indices):
            trajectory_indices = [(state_index + j) % self._replay_capacity
                                  for j in range(self._update_horizon)]
            trajectory_terminals = self._store['terminal'][trajectory_indices]
            is_terminal_transition = trajectory_terminals.any()
            if not is_terminal_transition:
                trajectory_length = self._update_horizon
            else:
                # np.argmax of a bool array returns the index of the first True.
                trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
                                              0) + 1
            next_state_index = state_index + trajectory_length
            trajectory_discount_vector = (
                self._cumulative_discount_vector[:trajectory_length])
            trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                                next_state_index)

            # Fill the contents of each array in the sampled batch.
            # assert len(transition_elements) == len(batch_arrays)
            for element_array, element in zip(batch_arrays, transition_elements):
                if element.name == 'observation':
                    element_array.append(self.get_observation_stack(state_index).squeeze(-1)) # remove the extra dim
                elif element.name == 'reward':
                    # compute the discounted sum of rewards in the trajectory.
                    element_array.append(np.sum(
                        trajectory_discount_vector * trajectory_rewards, axis=0))
                elif element.name == 'next_obs':
                    element_array.append(self.get_observation_stack(
                        (next_state_index) % self._replay_capacity).squeeze(-1))
                # elif element.name in ('next_action', 'next_reward'):
                #     element_array[batch_element] = (
                #         self._store[element.name.lstrip('next_')][(next_state_index) %
                #                                                   self._replay_capacity])
                elif element.name == 'terminal':
                    element_array.append(is_terminal_transition)
                elif element.name == 'indices':
                    element_array.append(state_index)
                elif element.name in self._store.keys():
                    element_array.append(self._store[element.name][state_index])
                # We assume the other elements are filled in by the subclass.

        return tuple([np.stack(x) for x in batch_arrays])

class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=20,
                    fc2_units=100, fc3_units=100, cat_sizes=tuple(),
                 use_wkday_feat=False, use_bn=False, position_encoding=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Network, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.position_encoding = position_encoding
        self.cat_embeddings = []
        if len(cat_sizes)>0:
            emb_sizes = [(size, min(600, round(1.6 * size ** 0.56))) for size in cat_sizes]
            self.cat_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes])

        if self.position_encoding>0: # 1: late addition 2: early fusion
            if self.position_encoding == 2:
                embsize = min(600, round(1.6*(action_size+1)**0.56))
            else: embsize = action_size
            self.action_slot_embedding = nn.Embedding(action_size+1, embsize)

        self.cat_dim = len(cat_sizes)
        emb_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        self.use_wkday_feat = use_wkday_feat
        activation = nn.LeakyReLU()
        mlp_in = state_size if self.cat_dim == 0 else state_size - self.cat_dim + emb_dim

        if use_wkday_feat:
            bn_in = mlp_in - 2
        else: bn_in = mlp_in - 1
        self.bn = nn.BatchNorm1d(bn_in)

        self.bn_in = bn_in
        self.use_bn = use_bn
        if self.position_encoding == 2:
            mlp_in += embsize
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, fc1_units),
            activation,
            nn.BatchNorm1d(fc1_units),
            nn.Linear(fc1_units, fc2_units),
            activation,
            # nn.BatchNorm1d(fc2_units),
            # nn.Linear(fc2_units, fc3_units),
            # activation,
            nn.BatchNorm1d(fc3_units),
            nn.Linear(fc3_units, action_size)
        )
        # self.fc1 = nn.Linear(state_size if self.cat_dim==0 else state_size-self.cat_dim+emb_dim, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.fc4 = nn.Linear(fc3_units, action_size)

    def init_embeddings(self, model_path):
        model = torch.load(model_path)
        model_sd = model.state_dict()
        embeddings = model_sd['embeddings']

    def get_device(self):
        return self.mlp[0].weight.device

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        device = self.get_device()
        if state.ndim==1: state = state[None] # expand first dim
        if self.cat_embeddings :
            catfeat = torch.LongTensor(state[...,:self.cat_dim]).to(device)
            confeat = torch.FloatTensor(state[...,self.cat_dim:]).to(device)
            embs = [e(catfeat[:,i]) for i,e in enumerate(self.cat_embeddings)]
            embs.append(confeat)
            state = torch.cat(embs, dim=1)
        else:
            catfeat = None
            if self.position_encoding>0:
                catfeat = torch.LongTensor(state[..., -1].astype(int)).to(device)
                state = torch.FloatTensor(state[..., :-1]).to(device)

            state = torch.FloatTensor(state).to(device)
            if self.use_bn:
                bn_part = self.bn(state[:,-self.bn_in:])
            else: bn_part = state[:, -self.bn_in:]
            to_concat = [state[:,:-self.bn_in], bn_part]
            if self.position_encoding==2:
                encoding = self.action_slot_embedding(catfeat)
                to_concat.append(encoding)
            state = torch.cat(to_concat, dim=1)
        # x = F.LeakyRelu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        output = self.mlp(state)
        if self.position_encoding==1:
            encoding = self.action_slot_embedding(catfeat)
            output = output+encoding
        return output

class StateNormalizer(nn.Module):
    def __init__(self, cfg, state_size, use_bn, value_in_state, use_history_encoder=False, encoder_type='gaussian', history_size=6, history_emb_dim=10,
                 history_type='sar',use_history_decoder=False, use_curiosity=False, reconstruction_num=0):
        super(StateNormalizer, self).__init__()
        self.encoder_type = encoder_type
        self.history_size = history_size
        self.reconstruct_and_modify = reconstruction_num > 0 and use_history_decoder
        self.reconstruction_num = reconstruction_num
        self.cat_dim = 0
        self.kl_factor = eval(cfg['rl_agent']['kl_factor'])
        emb_dim = 0
        state_in = state_size if self.cat_dim == 0 else state_size - self.cat_dim + emb_dim

        # if use_wkday_feat:
        #     bn_in = bn_in - 2
        # else: bn_in = bn_in - 1
        self.bn_in = state_in - 1
        self.bn = nn.BatchNorm1d(self.bn_in)
        if value_in_state: # cannot backprop
            self.init_value = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        else: self.init_value = None

        self.use_bn = use_bn
        # self.use_history_encoder = use_history_encoder
        if use_history_encoder:
            self.sar_size = (state_size + 1 + 1) if history_type=='sar' else state_size*2+2
            if encoder_type == 'gaussian':
                self.history_encoder = GaussianEncoder([200, 200], output_size=history_emb_dim, input_size=self.sar_size) # sar*history_size
            elif encoder_type == 'lstm':
                self.history_encoder = LSTMEncoder([200, 200], output_size=history_emb_dim,
                                                       input_size=self.sar_size)
            elif encoder_type == 'bilstm':
                self.history_encoder = LSTMEncoder([200, 200], output_size=history_emb_dim,
                                               input_size=self.sar_size, bidirectional=True)
            elif encoder_type == 'transformer':
                self.history_encoder = BertEncoder(cfg, input_size=self.sar_size, max_position=history_size, output_size=history_emb_dim, )
        else: self.history_encoder = None
        if use_history_decoder:
            if self.reconstruct_and_modify: decoder_target_dim = reconstruction_num
            else:
                decoder_target_dim = 1
            if encoder_type == 'transformer':
                self.history_decoder = BertDecoder(cfg, output_size=decoder_target_dim, )
            else: self.history_decoder = MLPDecoder([200, 200], output_size=decoder_target_dim, input_size=history_emb_dim)
            self.reconstruction = nn.MSELoss(reduction='none')
        else: self.history_decoder = None
        if use_curiosity:
            # pred_inp_dim = state_size + 1 # state + action
            # self.icm = MLPDecoder([200, 200], output_size=2, input_size=pred_inp_dim) # [t, ratio, Lsat] [slot_Lsat, slot_rev, profit]
            # self.reconstruction = nn.MSELoss()
            self.icm = ICM(state_size)
        else: self.icm = None
        self.bn = nn.BatchNorm1d(self.bn_in) if self.use_bn else nn.Identity()

    def forward_curiosity_loss(self, states, actions, next_states):
        # inp = torch.cat([states, actions.reshape((-1,1))], dim=-1)
        # pred_states = self.icm(inp)
        # curloss = self.reconstruction(pred_states, next_states[:,2:4])
        diff_loss, action_loss = self.icm.forward_loss(states, actions, next_states)
        return diff_loss+action_loss

    def forward(self, state, device, kl_div=False, is_deterministic=False, next_state_or_target_state=None):
        ret = dict()
        if self.history_encoder: # not None, then input is different
            state, history, num_history = state # history: padded tensor, num_history: list of nums
        else: history, num_history = None, None
        if state.ndim==1:
            state_ = state[None] # expand first dim
            if history is not None:
                # history = history[None]
                num_history = [num_history]
        else: state_ = state

        state = torch.from_numpy(state_).to(device)
        if self.history_encoder:

            if self.encoder_type=='transformer':
                nseq = self.history_size
                if len(num_history)==1: #
                    ns = 1
                    padded_hist = np.pad(history, ((0,nseq-history.shape[0]),(0,0)), constant_values=0, mode='constant') # 1,nhistory*state_size
                    history = torch.from_numpy(padded_hist.reshape((1,nseq,-1))).to(device) # reshape to 1,nhistory, state_size
                else:
                    ns = history.shape[0] # reshape to bs,nseq,dim
                    history = torch.from_numpy(history.reshape((ns,nseq,-1))).to(device)  # note that history uses the reversed order
                    ns = history.shape[0]
                nseq = self.history_size
                masks = np.ones((ns, nseq), dtype=np.float32)
                # 0 used for padding
                pos = np.zeros((ns, nseq), dtype=np.long)
                for idx, num in enumerate(num_history):
                    num_clipped = max(num,1) # todo: currently use zero padding to obtain initial distribution
                    if num<nseq:
                        masks[idx][num_clipped:] = 0
                    pos[idx][:num] = np.arange(num, 0, -1) # no need to use 1 for zero padding
                pos = torch.from_numpy(pos).to(device)
                masks = torch.from_numpy(masks).to(device)
                tmp = self.history_encoder.forward(history, pos, masks, kl_div=kl_div, is_deterministic=is_deterministic)
            else:
                history = torch.from_numpy(history).to(device)  # note that history uses the reversed order
                tmp = self.history_encoder.forward(history, num_history, kl_div=kl_div,
                                                   is_deterministic=is_deterministic)
            if kl_div:
                encodings, hidden, kl_loss = tmp
                ret['kl_loss'] = kl_loss*self.kl_factor
            else: encodings, hidden = tmp
            if self.reconstruct_and_modify: # decoder is guranteed
                if self.encoder_type=='transformer':
                    pred = self.history_decoder(hidden, masks)
                else: pred = self.history_decoder(encodings)
                # reconstruction
                if next_state_or_target_state is not None: # target_state
                    gold = state[:, -self.reconstruction_num:]
                    mse = self.reconstruction(pred, gold) # num, recnum

                    ret['mse_loss'] = mse.mean(0).sum()
                # replace
                state = torch.cat((state[:,:-self.reconstruction_num], pred.detach()), dim=-1)
            else:
                if next_state_or_target_state is not None and self.history_decoder is not None:
                    pred = self.history_decoder(encodings)
                    gold = torch.from_numpy(next_state_or_target_state[0][:, 2:-2]).to(device)
                    mse = self.reconstruction(pred, gold)
                    ret['mse_loss'] = mse

            # ret['z'] = encodings
            # action = torch.FloatTensor(action).to(device)
        if self.init_value is not None:
            selector = state[:, 0] == 0.
            # num = ((state_[:,0])<=1e-4).sum()
            state[:,0] += self.init_value*selector
        if self.use_bn:
            bn_part = self.bn(state[:, -self.bn_in:])
            to_concat = [state[:,:-self.bn_in], bn_part]
            mlp_inp = torch.cat(to_concat, dim=1)
        else:
            mlp_inp = state
        ret['state'] = mlp_inp
        if self.history_encoder:
            ret['state'] = torch.cat([mlp_inp, encodings], dim=-1)

        return ret

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def _product_of_gaussians(mus, sigmas_squared, sigmas_squared_reciprocal):
    '''
    compute mu, sigma of product of gaussians
    '''
    # sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(sigmas_squared_reciprocal, dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

class MLP(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            # output_size,
            input_size,
            hidden_activation=F.relu,
            # output_activation=nn.Identity(),
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        # self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        # self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_size = self.hidden_sizes[-1]
        self.hidden_activation = hidden_activation
        # self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

    def forward(self, h):
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            # if self.layer_norm and i < len(self.fcs) - 1:
            #     h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        return h

class MLPEncoder(MLP):
    def forward(self, history, num_history):
        """
        history: bs, multiples of sar_size
        num_history: list of ints
        """
        inp = []
        if len(num_history)==1: # from policy
            h = history
            # sample = history[0]
            # h = sample.reshape((max(num_history[0],1), -1))
        else:
            # bs, nseq_dim = history.shape
            history = history.reshape(len(num_history), -1, self.input_size)
            for idx,num in enumerate(num_history):
                sample = history[idx]
                inp.append(sample[:max(num,1)]) # todo: when there is only padded, use zero padding feature to obtain initial parameters
            h = torch.cat(inp, dim=0)

        return super(MLPEncoder, self).forward(h)

class GaussianEncoder(MLPEncoder):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=nn.Identity(),
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        # self.save_init_params(locals())
        super(GaussianEncoder, self).__init__(
            hidden_sizes,
            # output_size,
            input_size,
            hidden_activation,
            # output_activation,
            hidden_init,
            b_init_value,
            layer_norm,
            layer_norm_kwargs,)

        self.output_size = output_size
        self.output_activation = output_activation
        self.last_fc = nn.Linear(self.hidden_size, output_size*2) # for mean and variance
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, history, num_history, kl_div=False, is_deterministic=False):
        """
        history: bs, multiples of sar_size
        num_history: list of ints
        """
        h = super(GaussianEncoder, self).forward(history, num_history)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        ### get x, outdim
        mu = output[..., :self.output_size]
        sigma_squared = F.softplus(output[..., self.output_size:])
        sigmas_squared = torch.clamp(sigma_squared, min=1e-7)
        sigma_squared_reciprocal = torch.reciprocal(sigmas_squared)
        z_means, z_vars = [],[]
        cnt = 0
        for num in num_history:
            m,s,sr = mu[cnt:cnt+max(num,1)], sigmas_squared[cnt:cnt+max(num,1)], sigma_squared_reciprocal[cnt:cnt+max(num,1)]
            p = _product_of_gaussians(m, s, sr)
            z_means.append(p[0])
            z_vars.append(p[1])

        # z_distrs = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
        #            zip(z_means, z_vars)]
        #
        # z = torch.stack([dist.rsample() for dist in z_distrs])
        z_means = torch.stack(z_means)
        z_distrs = torch.distributions.Normal(z_means, torch.sqrt(torch.stack(z_vars))) # todo: check if this sqrt leads to any bad things
        z = z_distrs.rsample()
        if is_deterministic: z = z_means

        if kl_div:
            device = h.device
            # prior = torch.distributions.Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))
            # kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in z_distrs]
            # kl_div_mean = torch.mean(torch.stack(kl_divs))
            prior = torch.distributions.Normal(torch.zeros_like(z).to(device), torch.ones_like(z).to(device))
            kl_divs = torch.distributions.kl.kl_divergence(z_distrs, prior)
            kl_div_mean = torch.mean(kl_divs.sum(-1))

            return z, None, kl_div_mean
        else: return z, None

class LSTMEncoder(MLP):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=nn.Identity(),
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            bidirectional=False,
            num_lstm=1,
    ):
        super(LSTMEncoder, self).__init__(
            hidden_sizes,
            # output_size,
            input_size,
            hidden_activation,
            # output_activation,
            hidden_init,
            b_init_value,
            layer_norm,
            layer_norm_kwargs, )

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_lstm, batch_first=True, bidirectional=bidirectional)
        self.output_size = output_size
        self.output_activation = output_activation
        self.last_fc = nn.Linear(self.hidden_size*2 if bidirectional else self.hidden_size, output_size * 2)  # for mean and variance
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, history, num_history, kl_div=False, is_deterministic=False): # history is in the reverse order
        history = history.reshape(len(num_history), -1, self.input_size) # bs,nseq,dim
        init = torch.zeros_like(history).to(history.device)
        for idx,num in enumerate(num_history):
            floored_num = max(num,1)
            init[idx,:num] = torch.flip(history[idx,:floored_num], dims=[-1]) # reverse

        h = super(LSTMEncoder, self).forward(init)
        bs,nseq,dim = h.shape
        # hidden = (torch.zeros(bs, 1, dim), torch.zeros(bs, 1, dim)) # nseq=1 is enough
        out, hidden = self.lstm(h) # out: bs,nseq,dim
        final = []
        for idx,num in enumerate(num_history):
            floored_num = max(num, 1)
            final.append(out[idx,floored_num-1])
        final = torch.stack(final) # bs,dim
        preactivation = self.last_fc(final)
        output = self.output_activation(preactivation)

        mu = output[..., :self.output_size]  # bs,dim
        sigma_squared = F.softplus(output[..., self.output_size:])
        sigmas = torch.sqrt(torch.clamp(sigma_squared, min=1e-7))
        z_means = mu  # torch.stack(mu)
        z_distrs = torch.distributions.Normal(z_means, sigmas)  # todo: check if this sqrt leads to any bad things
        z = z_distrs.rsample()
        if is_deterministic: z = z_means

        if kl_div:
            device = out.device
            # prior = torch.distributions.Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))
            # kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in z_distrs]
            # kl_div_mean = torch.mean(torch.stack(kl_divs))
            prior = torch.distributions.Normal(torch.zeros_like(z).to(device), torch.ones_like(z).to(device))
            kl_divs = torch.distributions.kl.kl_divergence(z_distrs, prior)
            kl_div_mean = torch.mean(kl_divs.sum(-1))

            return z, out, kl_div_mean
        else:
            return z, out


class BertEncoder(nn.Module):
    def __init__(self, config, input_size, max_position, output_size, output_activation=nn.Identity(), init_w=3e-3,):
        super().__init__()
        use_layer = eval(config['bert']['num_layer'])
        hidden_size = eval(config['bert']['hidden_size'])
        hidden_dropout_prob = eval(config['bert']['hidden_dropout_prob'])
        self.bidirection = eval(config['bert']['bidirection']) # affect attention mask
        self.bert_pooler = config['bert']['bert_pooler']
        layer = BertLayer(config) # CONFIG WILL be turned into dict
        self.inp_embedding = BertEmbedding(input_size, hidden_size, max_position, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(use_layer)])
                                    # for _ in range(config.num_hidden_layers)])
        self.output_size = output_size
        self.output_activation = output_activation
        self.last_fc = nn.Linear(hidden_size, output_size * 2)  # for mean and variance
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input_, pos, attention_mask,
                kl_div=False, is_deterministic=False
                ):
        """
        input_: padded tensor, bs,nseq,dim
        attention_mask: bs,nseq, pad locations are labeled 0
        """
        # all_encoder_layers = []
        # all_layer_atts=[]
        nseq = attention_mask.shape[1]
        hidden_states = self.inp_embedding(input_, pos) # batchsize, nseq, dim
        if self.bidirection==1: # if use bidrection:
            # padded ones will be 1
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # expand dim for num_head and attention
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else: # only use upper triangular
            # mask out future
            masks = []
            for seqlen in attention_mask.sum(-1).cpu().numpy(): # nparray of bs
                seqlen = int(seqlen)
                unit_upper_tri = np.triu(np.ones((seqlen, seqlen), dtype=np.float32))
                init = np.zeros((nseq,nseq), dtype=np.float32)
                init[:seqlen, :seqlen] = unit_upper_tri[:,::-1] # reverse order
                masks.append(init)
            extended_attention_mask = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(attention_mask.device) # bs,nseq,nseq

        for layer_module in self.layer:
            hidden_states,attention_probs = layer_module(hidden_states, extended_attention_mask)

        if self.bert_pooler == 'mean':
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / torch.sum(attention_mask, dim=1,
                                                                                       keepdim=True)  # bs, dim
        else:  # first token
            pooled = hidden_states[:, 0]


        preactivation = self.last_fc(pooled)
        output = self.output_activation(preactivation)
        mu = output[..., :self.output_size] # bs,dim
        sigma_squared = F.softplus(output[..., self.output_size:])
        sigmas = torch.sqrt(torch.clamp(sigma_squared, min=1e-7))
        z_means = mu #torch.stack(mu)
        z_distrs = torch.distributions.Normal(z_means, sigmas)  # todo: check if this sqrt leads to any bad things
        z = z_distrs.rsample()
        if is_deterministic: z = z_means

        if kl_div:
            device = hidden_states.device
            # prior = torch.distributions.Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))
            # kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in z_distrs]
            # kl_div_mean = torch.mean(torch.stack(kl_divs))
            prior = torch.distributions.Normal(torch.zeros_like(z).to(device), torch.ones_like(z).to(device))
            kl_divs = torch.distributions.kl.kl_divergence(z_distrs, prior)
            kl_div_mean = torch.mean(kl_divs.sum(-1))

            return z, hidden_states, kl_div_mean
        else: return z, hidden_states


class MLPDecoder(MLP):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=nn.LeakyReLU(),
            output_activation=nn.Identity(),
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        # self.save_init_params(locals())
        super(MLPDecoder, self).__init__(
            hidden_sizes,
            # output_size,
            input_size,
            hidden_activation,
            # output_activation,
            hidden_init,
            b_init_value,
            layer_norm,
            layer_norm_kwargs,
        )

        self.output_activation = output_activation
        self.last_fc = nn.Linear(self.hidden_size, output_size) # for mean and variance
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, inp):
        """
        history: bs, multiples of sar_size
        num_history: list of ints
        """
        h = super(MLPDecoder, self).forward(inp)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output

class BertDecoder(nn.Module):
    def __init__(self, config, output_size, output_activation=nn.Identity(), init_w=3e-3,):
        super().__init__()
        use_layer = eval(config['bert']['num_layer'])
        hidden_size = eval(config['bert']['hidden_size'])
        # hidden_dropout_prob = eval(config['bert']['hidden_dropout_prob'])
        self.bidirection = eval(config['bert']['bidirection'])  # affect attention mask
        self.bert_pooler = config['bert']['bert_pooler']
        layer = BertLayer(config) # CONFIG WILL be turned into dict
        # self.inp_embedding = BertEmbedding(input_size, hidden_size, max_position, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(use_layer)])
                                    # for _ in range(config.num_hidden_layers)])
        self.output_size = output_size
        self.output_activation = output_activation
        self.last_fc = nn.Linear(hidden_size, output_size)  # for mean and variance
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, hidden_states, attention_mask):
        """
        input_: padded tensor, bs,nseq,dim
        attention_mask: bs,nseq, pad locations are labeled 0
        """

        nseq = attention_mask.shape[1]
        if self.bidirection == 1:  # if use bidrection:
            # padded ones will be 1
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # expand dim for num_head and attention
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:  # only use upper triangular
            # mask out future
            masks = []
            for seqlen in attention_mask.sum(-1).cpu().numpy():  # nparray of bs
                seqlen = int(seqlen)
                unit_upper_tri = np.triu(np.ones((seqlen, seqlen), dtype=np.float32))
                init = np.zeros((nseq, nseq), dtype=np.float32)
                init[:seqlen, :seqlen] = unit_upper_tri[:, ::-1]  # reverse order
                masks.append(init)
            extended_attention_mask = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(
                attention_mask.device)  # bs,nseq,nseq

        for layer_module in self.layer:
            hidden_states,attention_probs = layer_module(hidden_states, extended_attention_mask)

        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / torch.sum(attention_mask, dim=1,
                                                                                   keepdim=True)  # bs, dim

        preactivation = self.last_fc(pooled)
        output = self.output_activation(preactivation)
        return output

class BertEmbedding(nn.Module):
    def __init__(self, input_size, output_size, max_position, hidden_dropout_prob):
        super(BertEmbedding, self).__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, output_size),
            # nn.LayerNorm(output_size, eps=1e-12)
        )

        self.position_embedding = nn.Embedding(max_position+1, output_size, padding_idx=0) # pad at 0, idx is the order
        self.LayerNorm = nn.LayerNorm(output_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, inp, pos):
        inp = self.feature_transform(inp)
        emb = inp + self.position_embedding(pos)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        return emb

class StateActionValueNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=20,
                    fc2_units=100, fc3_units=100, cat_sizes=tuple(),
                 use_wkday_feat=False, use_bn=False, position_encoding=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(StateActionValueNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.position_encoding = position_encoding
        self.cat_embeddings = []
        self.cat_dim = 0

        emb_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        self.use_wkday_feat = use_wkday_feat
        activation = nn.LeakyReLU()
        state_in = state_size if self.cat_dim == 0 else state_size - self.cat_dim + emb_dim
        self.use_bn = use_bn
        self.bn_in = state_in-1 if self.use_bn else 0 # if

        mlp_in = state_in + action_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, fc1_units),
            activation,
            # nn.BatchNorm1d(fc1_units),
            nn.Linear(fc1_units, fc2_units),
            activation,
            # nn.BatchNorm1d(fc2_units),
            # nn.Linear(fc2_units, fc3_units),
            # activation,
            # nn.BatchNorm1d(fc3_units),
            nn.Linear(fc3_units, 1)
        )
        self.possible_initialization()

    def possible_initialization(self):
        # uniform init last output layer
        pass

    def init_embeddings(self, model_path):
        model = torch.load(model_path)
        model_sd = model.state_dict()
        embeddings = model_sd['embeddings']

    def get_device(self):
        return self.mlp[0].weight.device

    def set_normalizer(self, normalizer):
        if self.use_bn:
            self.state_normalizer = normalizer

    def forward(self, state, action): # todo: action may need normalization
        """Build a network that maps state -> action values."""
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # device = self.get_device()
        if action.ndim==1:
            # state = state[None] # expand first dim
            action = action[None]

        mlp_inp = torch.cat([state, action.reshape((-1,1))], dim=-1)

        output = self.mlp(mlp_inp)

        return output

class DoubleQCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_hidden=20, tau=0.01,
                 fc2_units=100, fc3_units=100, cat_sizes=tuple(),
                 use_wkday_feat=False, use_bn=False, position_encoding=0):
        super(DoubleQCritic, self).__init__()
        self.use_bn = use_bn
        self.tau = tau
        self.q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)
        self.q2 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)
        self.target_q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden,
                                                 use_bn=self.use_bn)
        self.target_q2 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden,
                                                 use_bn=self.use_bn)
        for p in self.target_q2.parameters():
            p.requires_grad_(False)

        for p in self.target_q1.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.sync_weights(syn_method='copy')

        self.q1_criterion, self.q2_criterion = nn.MSELoss(), nn.MSELoss()

        self.networks = [self.q1, self.q2, self.target_q1, self.target_q2]

        self.auto_entropy = True
        self.target_entropy = -action_size

    def set_normalizer(self, normalizer):
        [x.set_normalizer(normalizer) for x in self.networks]

    def target_qv(self, next_states, next_action, rewards, gamma, dones):
        next_qv = torch.min(self.target_q1(next_states, next_action), self.target_q2(next_states, next_action))
        target_qv = rewards.unsqueeze(-1) + (gamma * next_qv * (1. - dones.unsqueeze(-1).float()))
        return next_qv, target_qv

    def new_qv(self, states, new_action):
        new_qv = torch.min(self.q1(states, new_action), self.q2(states, new_action))
        return new_qv

    def forward_loss(self, states, actions, new_action_logp, target_qv):
        qv1 = self.q1(states, actions)
        qv2 = self.q2(states, actions)

        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (new_action_logp + self.target_entropy).detach()).mean()
            # self.alpha_opt.zero_grad()
            # alpha_loss.backward()
            # self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0.

        q1_loss = self.q1_criterion(qv1, target_qv.detach())
        q2_loss = self.q2_criterion(qv2, target_qv.detach())
        qloss = (q1_loss + q2_loss) / 2.

        return qloss, alpha_loss

    def sync_weights(self, syn_method='avg'):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        sync_weights(self.q1, self.target_q1, self.tau, syn_method=syn_method)
        sync_weights(self.q2, self.target_q2, self.tau, syn_method=syn_method)

class DoubleQCriticMC(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_hidden=20, tau=0.01,
                 fc2_units=100, fc3_units=100, cat_sizes=tuple(),
                 use_wkday_feat=False, use_bn=False, position_encoding=0, auto_entropy=True):
        super(DoubleQCriticMC, self).__init__()
        self.use_bn = use_bn
        self.tau = tau
        self.q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)
        self.q2 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)
        # no target since directly use the monte carlo result
        # self.target_q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden,
        #                                          use_bn=self.use_bn)
        # self.target_q2 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden,
        #                                          use_bn=self.use_bn)
        # for p in self.target_q2.parameters():
        #     p.requires_grad_(False)
        #
        # for p in self.target_q1.parameters():
        #     p.requires_grad_(False)

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.sync_weights(syn_method='copy')

        self.q1_criterion, self.q2_criterion = nn.MSELoss(), nn.MSELoss()

        self.networks = [self.q1, self.q2]

        self.auto_entropy = auto_entropy
        self.target_entropy = -action_size

    def set_normalizer(self, normalizer):
        [x.set_normalizer(normalizer) for x in self.networks]

    # def target_qv(self, next_states, next_action, rewards, gamma, dones):
    #     next_qv = torch.min(self.q1(next_states, next_action), self.q2(next_states, next_action))
    #     target_qv = rewards.unsqueeze(-1) + (gamma * next_qv * (1 - dones.unsqueeze(-1)))
    #     return next_qv, target_qv

    def new_qv(self, states, new_action):
        new_qv = torch.min(self.q1(states, new_action), self.q2(states, new_action))
        return new_qv

    def forward_loss(self, states, actions, new_action_logp, target_qv):
        qv1 = self.q1(states, actions)
        qv2 = self.q2(states, actions)

        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (new_action_logp + self.target_entropy).detach()).mean()
            # self.alpha_opt.zero_grad()
            # alpha_loss.backward()
            # self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = torch.tensor(0.).to(self.q1.get_device())

        q1_loss = self.q1_criterion(qv1.reshape(-1), target_qv.detach())
        q2_loss = self.q2_criterion(qv2.reshape(-1), target_qv.detach())
        qloss = (q1_loss + q2_loss) / 2.

        return qloss, alpha_loss

    def sync_weights(self, syn_method='avg'): # no need to sync weight since directly use target
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # sync_weights(self.q1, self.target_q1, self.tau, syn_method=syn_method)
        # sync_weights(self.q2, self.target_q2, self.tau, syn_method=syn_method)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_hidden=20, tau=0.01,
                 fc2_units=100, fc3_units=100, cat_sizes=tuple(),
                 use_wkday_feat=False, use_bn=False, position_encoding=0, auto_entropy=True):
        super(Critic, self).__init__()
        self.use_bn = use_bn
        self.tau = tau
        self.q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)
        # self.q2 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)
        # no target since directly use the monte carlo result
        self.target_q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden,
                                                 use_bn=self.use_bn)

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.sync_weights(syn_method='copy')

        self.q1_criterion = nn.MSELoss()

        self.networks = [self.q1]


    def set_normalizer(self, normalizer):
        [x.set_normalizer(normalizer) for x in self.networks]

    def new_qv(self, states, new_action):
        return self.q1(states, new_action)

    def target_qv(self, next_states, next_action, rewards, gamma, dones):
        next_qv = self.target_q1(next_states, next_action)
        target_qv = rewards.unsqueeze(-1) + (gamma * next_qv * (1. - dones.unsqueeze(-1).float()))
        return next_qv, target_qv

    def forward_loss(self, states, actions, new_action_logp, target_qv):
        qv1 = self.q1(states, actions)

        q1_loss = self.q1_criterion(qv1, target_qv.detach())

        return q1_loss, torch.tensor(0.).to(self.q1.get_device())

    def sync_weights(self, syn_method='avg'): # no need to sync weight since directly use target
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        pass

class CriticMC(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_hidden=20, tau=0.01,
                 fc2_units=100, fc3_units=100, cat_sizes=tuple(),
                 use_wkday_feat=False, use_bn=False, position_encoding=0, auto_entropy=True):
        super(CriticMC, self).__init__()
        self.use_bn = use_bn
        self.tau = tau
        self.q1 = StateActionValueNetwork(state_size, action_size, seed, fc1_units=fc1_hidden, use_bn=self.use_bn)


        self.log_alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.sync_weights(syn_method='copy')

        self.q1_criterion = nn.MSELoss()

        self.networks = [self.q1]


    def set_normalizer(self, normalizer):
        [x.set_normalizer(normalizer) for x in self.networks]


    def new_qv(self, states, new_action):
        return self.q1(states, new_action)

    def forward_loss(self, states, actions, new_action_logp, target_qv):
        qv1 = self.q1(states, actions)

        q1_loss = self.q1_criterion(qv1.reshape(-1), target_qv.detach())

        return q1_loss, torch.tensor(0.).to(self.q1.get_device())

    def sync_weights(self, syn_method='avg'): # no need to sync weight since directly use target
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        pass

class PolicyNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, action_bound=(-1,1), fc1_units=20,
                    fc2_units=100, fc3_units=100, cat_sizes=tuple(), use_bn=False,
                 use_wkday_feat=False, position_encoding=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PolicyNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.position_encoding = position_encoding
        self.cat_embeddings = []
        self.cat_dim = 0
        self.noise_std = 0.05

        self.action_scale = (action_bound[1] - action_bound[0])/2.
        self.action_center = (action_bound[1]+action_bound[0])/2.
        emb_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        self.use_wkday_feat = use_wkday_feat
        activation = nn.LeakyReLU()
        state_in = state_size if self.cat_dim == 0 else state_size - self.cat_dim + emb_dim
        self.use_bn = use_bn
        self.bn_in = state_in-1 if self.use_bn else 0

        mlp_in = state_in
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, fc1_units),
            activation,
            # nn.BatchNorm1d(fc1_units),
            nn.Linear(fc1_units, fc2_units),
            activation,
            # nn.BatchNorm1d(fc2_units),
            # nn.Linear(fc2_units, fc3_units),
            # activation,
            # nn.BatchNorm1d(fc3_units),
        )
        self.mean_fc = nn.Linear(fc2_units, action_size)
        self.logstd_fc = nn.Linear(fc2_units, action_size)

    def init_embeddings(self, model_path):
        model = torch.load(model_path)
        model_sd = model.state_dict()
        embeddings = model_sd['embeddings']

    def get_device(self):
        return self.mlp[0].weight.device

    def set_normalizer(self, normalizer):
        if self.use_bn:
            self.state_normalizer = normalizer

    def forward(self, state):

        mlp_inp = state
        hidden = self.mlp(mlp_inp)
        mean = self.mean_fc(hidden)
        logstd = self.logstd_fc(hidden)
        logstd = torch.clamp(logstd, min=LOGSTD_MIN, max=LOGSTD_MAX)

        # if self.position_encoding==1:
        #     encoding = self.action_slot_embedding(catfeat)
        #     output = output+encoding
        return mean, logstd

    def distr_params(self, state):
        mean, logstd = self.forward(state)
        std = logstd.exp()
        sampler = Normal(mean, std)
        entropy = sampler.entropy()
        scaled_mean = torch.tanh(mean) * self.action_scale + self.action_center
        return mean, scaled_mean, std, sampler, entropy

    def sample(self, state, keep_params=False):
        # https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
        mean, scaled_mean, std, sampler, entropy = self.distr_params(state)
        before = sampler.rsample() # reparameterization trick
        after = torch.tanh(before) # squeeze to -1,1
        logp = sampler.log_prob(before)
        # enforce bound
        logp -= torch.log((1.-after.pow(2))*self.action_scale + EPS)
        logp = logp.sum(1, keepdim=True) # -1,1

        sampled_action = after * self.action_scale + self.action_center
        if not keep_params:
            return sampled_action, logp
        else:
            return sampled_action, logp, mean, entropy

    def deterpolicy_action(self, state, is_deterministic=False):
        mean, scaled_mean, std, sampler, entropy = self.distr_params(state)

        action = scaled_mean
        if not is_deterministic:
            action_noise_std = self.noise_std
            eps = torch.randn_like(mean).to(mean.device) * action_noise_std
            sampled_action = eps + action
        else: sampled_action = action
        return sampled_action

    def sync_weights(self, syn_method='avg'): # no need to sync weight since directly use target
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        pass

class DoublePolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, action_bound=(-1, 1), fc1_units=20,
                 fc2_units=100, fc3_units=100, cat_sizes=tuple(), use_bn=False,
                 use_wkday_feat=False, position_encoding=0, tau=0.9):
        super(DoublePolicyNetwork, self).__init__()
        self.tau = tau
        self.imp_policy = PolicyNetwork(state_size, action_size, seed, action_bound, fc1_units,
                 fc2_units, fc3_units, cat_sizes, use_bn,
                 use_wkday_feat, position_encoding)
        self.target_policy = PolicyNetwork(state_size, action_size, seed, action_bound, fc1_units,
                                        fc2_units, fc3_units, cat_sizes, use_bn,
                                        use_wkday_feat, position_encoding)

    def deterpolicy_action(self, state, is_deterministic=False):
        return self.imp_policy.deterpolicy_action(state, is_deterministic)

    def deterpolicy_target_action(self, state, is_deterministic=False):
        return self.target_policy.deterpolicy_action(state, is_deterministic)

    def sync_weights(self, syn_method='avg'): # no need to sync weight since directly use target
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        sync_weights(self.imp_policy, self.target_policy, self.tau, syn_method=syn_method)

    def get_device(self): return self.imp_policy.get_device()

    # def forward(self, state): return self.imp_policy(state)



class ICM(nn.Module):
    def __init__(self, state_size, hidden_size=100, state_feat_size=20, action_size=1):
        super(ICM, self).__init__()
        self.emb_model = MLP([hidden_size,state_feat_size], state_size)
        state_action_dim = state_feat_size + action_size
        self.forward_model = MLP([hidden_size, state_feat_size], state_action_dim)
        self.inverse_model = MLP([hidden_size*2, action_size], state_feat_size*2) # input state+state
        self.mse = nn.MSELoss()

    def forward(self, states, actions, next_states):
        sfeats = self.emb_model(states)
        nsfeats = self.emb_model(next_states)

        reconstr_nsfeats = self.forward_model(torch.cat([sfeats, actions.reshape((-1,1))], dim=-1))
        diff = ((nsfeats-reconstr_nsfeats)**2).mean(dim=-1)
        return diff, sfeats, nsfeats

    def forward_loss(self, states, actions, next_states):
        diff, sfeats, nsfeats = self.forward(states, actions, next_states)
        diff_loss = diff.mean()
        reconstr_actions = self.inverse_model(torch.cat([sfeats, nsfeats], dim=-1))
        inverse_loss = self.mse(reconstr_actions, actions.reshape((-1,1)))
        return diff_loss, inverse_loss







