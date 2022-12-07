"""Replay buffer class to store the most recent experiences.

You do *NOT* need to modify any code here.
"""

import numpy as np
import random
import collections


class Buffer:
    """Deque object to store the most recent experiences."""

    def __init__(self, size, sample_size):
        """Initialize a new buffer instance.

        Args:
          size (int): maximum size of the buffer.
          sample_size (int): size of the randomly sampled batch of experiences.
        """
        self._size = int(size)
        self._sample_size = sample_size
        self._buffer = collections.deque(maxlen=self._size)

    def add(self, state, action, reward, next_state):
        """Add an experience tuple to the buffer.

        Args:
          state (ndarray): a numpy array corresponding to the env state
          action (ndarray): a numpy array corresponding to the action
          reward (ndarray): reward obtained from the env
          next_state (ndarray): numpy array corresponding to the next state
        """
        self._buffer.append((state, action, reward, next_state))

    def sample(self):
        """
        Randomly sample experiences from the replay buffer.

        Returns:
          (tuple): batch of experience (state, action, reward, next_state)
        """
        samples = self._buffer
        if len(self._buffer) >= self._sample_size:
            samples = random.sample(self._buffer, self._sample_size)

        state = np.reshape(
            np.array([arr[0] for arr in samples]), [len(samples), -1])
        action = np.array([arr[1] for arr in samples])
        reward = np.array([arr[2] for arr in samples])
        next_state = np.array(
            [arr[3] for arr in samples]).reshape(len(samples), -1)

        return state, action, reward, next_state
