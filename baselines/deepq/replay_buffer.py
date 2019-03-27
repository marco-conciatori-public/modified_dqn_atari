import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        # return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
        return obses_t, actions, rewards, obses_tp1, dones

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class LowerBoundReplayBuffer(ReplayBuffer):
    def __init__(self, size, gamma):
        """Create Lower Bound Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        See Also
        --------
        ReplayBuffer.__init__
        """
        super().__init__(size)
        self._episode_transitions = []
        self.gamma = gamma
        # self.free_indexes = []

    def add(self, obs_t, action, reward, new_obs, *unused_args):
        # if len(self.free_indexes) > 0:
        #     self._storage[self.free_indexes.pop()] = (obs_t, action, reward, new_obs, float(True))
        # else:
        super().add(obs_t, action, reward, new_obs, float(True))

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            try:
                obs_t, action, reward, new_obs, done = data
            except TypeError as e:
                print('Errore:', e)
                print('Esperienza (tupla):', data)
                print('indice esp:', i)
                print('puntatore ciclico:', self._next_idx)
                # print('free_indexes:', self.free_indexes)
                print('len(_storage):', len(self._storage))
                raise SystemExit

            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(new_obs, copy=False))
            dones.append(done)

        # return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
        return obses_t, actions, rewards, obses_tp1, dones

    # def compute_lb(self):
    #     index = len(self._episode_transitions) - 1
    #     cumulative_reward = 0
    #     got_reward = False
    #     while index >= 0:
    #         obs_t, action, reward, new_obs = self._episode_transitions[index]
    #         if reward > 0:
    #             got_reward = True
    #
    #         if got_reward:
    #             cumulative_reward = cumulative_reward * self.gamma + reward
    #             self.add(obs_t, action, cumulative_reward, new_obs)
    #         index -= 1
    #
    #     self._episode_transitions = []

    def compute_lb(self, q_values):
        index = len(self._episode_transitions) - 1
        cumulative_reward = 0
        got_reward = False
        while index >= 0:
            obs_t, action, reward, new_obs = self._episode_transitions[index]
            if reward > 0:
                got_reward = True

            if got_reward:
                cumulative_reward = cumulative_reward * self.gamma + reward
                if test_single_exp(action, cumulative_reward, q_values, obs_t):
                    self.add(obs_t, action, cumulative_reward, new_obs)
            index -= 1

        self._episode_transitions = []

    def sample(self, batch_size):
        indexes = []
        max_index = len(self._storage) - 1
        while len(indexes) < batch_size:
            temp_index = random.randint(0, max_index)
            # if temp_index not in self.free_indexes and temp_index not in indexes:
            if temp_index not in indexes:
                indexes.append(temp_index)
        return self._encode_sample(indexes)

    # def sample(self, batch_size):
    #     to_choose_indexes = []
    #     for i in range(len(self._storage) - 1):
    #         if i not in self.free_indexes:
    #             to_choose_indexes.append(i)
    #     indexes = random.sample(to_choose_indexes, batch_size)
    #     return self._encode_sample(indexes)

    def memorize_transition(self, obs_t, action, reward, new_obs):
        data = (obs_t, action, reward, new_obs)
        self._episode_transitions.append(data)

    # def remove_experiences(self, to_remove):
    #     for i in to_remove:
    #         self._storage[i] = None
    #         self.free_indexes.append(i)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def test(actions, lb_rewards, estimated_rewards):
    indexes = []
    to_remove = []
    for i in range(len(lb_rewards)):
        lb_rew = lb_rewards[i]
        estimated_rew = estimated_rewards[i]
        act = actions[i]
        estimated_rew = estimated_rew[act]

        if lb_rew > estimated_rew:
            indexes.append(i)
        else:
            to_remove.append(i)

    return indexes, to_remove


def test_single_exp(action, lb_reward, q_values, obs_t):
    estimated_reward = q_values(np.array([obs_t]))
    print('estim_rew_all_actions:', estimated_reward)
    estimated_reward = estimated_reward[action]
    print('estimated_reward:', estimated_reward)
    return lb_reward > estimated_reward
