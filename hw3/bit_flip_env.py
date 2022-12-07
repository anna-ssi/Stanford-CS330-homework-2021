"""Environment for bit flipping.

You do *NOT* need to modify the code here.
"""

import numpy as np


class BitFlipEnv:
    """Bit flipping environment for reinforcement learning.
    The environment is a 1D vector of binary values (state vector).
    At each step, the actor can flip a single bit (0 to 1 or 1 to 0).
    The goal is to flip bits until the state vector matches the
    goal vector (also a 1D vector of binary values). At each step,
    the actor receives a reward of -1 if the state and goal vector
    do not match and a reward of 0 if the state and goal vector
    match.

    """

    def __init__(self, num_bits, verbose=False):
        """Initialize new instance of BitFlip class.

        Args:
            num_bits (int): number of bits in the environment
            verbose (bool): prints state and goal vector after each
                            step if True
        """

        # check that num_bits is a positive integer
        if num_bits < 0:
            raise ValueError("Invalid number of bits: must be positive integer")

        # number of bits in the environment
        self._num_bits = num_bits
        # randomly set the state vector
        self._state_vector = np.random.randint(0, 2, num_bits)
        # randomly set the goal vector
        self._goal_vector = np.random.randint(0, 2, num_bits)
        # whether to print debugging info
        self._verbose = verbose
        # set dimensions of observation space
        self._observation_space = self._state_vector
        # create action space; may use gym type
        self._action_space = num_bits
        # space of the goal vector
        self._goal_space = self._goal_vector

    def show_goal(self):
        """Returns the goal as a numpy array. Used for debugging.

        Returns:
            (ndarray): a numpy array of size (num_bits,)
        """
        return self._goal_vector.copy()

    def show_state(self):
        """Returns the state as a numpy array. Used for debugging.

        Returns:
            (ndarray): a numpy array of size (num_bits,)
        """
        return self._state_vector.copy()

    def reset(self):
        """Resets the environment.
        Returns:
            (ndarray): a state_vector of size (num_bits,)
            (ndarray): a goal vector of size (num_bits,)
        """
        # randomly reset both the state and the goal vectors
        self._state_vector = np.random.randint(
            0, 2, self._num_bits).astype(np.float32)
        self._goal_vector = np.random.randint(
            0, 2, self._num_bits).astype(np.float32)

        # return as numpy array
        return self._state_vector.copy(), self._goal_vector.copy()

    def step(self, action):
        """Take a step and flip one of the bits.

        Args:
            action (int): action in [0, num_bits - 1]

        Returns:
            state (ndarray): new state_vector of size (num_bits,)
            reward (float): 0 if state != goal and 1 if state == goal
            done (bool): value indicating if the goal has been reached
            info (dict): dictionary with the goal state and success boolean
        """

        info = {"goal_vector": self._goal_vector}

        if action < 0 or action >= self._num_bits:
            # check argument is in range
            raise ValueError(
                "Invalid action! Must be integer ranging from \
                0 to num_bits-1"
            )

        # flip the bit with index action
        self._state_vector[action] =  1 - self._state_vector[action]

        # check if state and goal vectors are identical
        if (self._state_vector == self._goal_vector).all():
            reward = 0
            info["successful_this_state"] = True
            done = True
        else:
            reward = -1
            info["successful_this_state"] = False
            done = False

        # print additional info if verbose mode is on
        if self._verbose:
            print("Bit flipped:   ", action)
            print("Goal vector:   ", self._goal_vector)
            print("Updated state: ", self._state_vector)
            print("Reward:        ", reward)

        return (
            np.array(self._state_vector).astype(np.float32),
            np.array(reward).astype(np.float32),
            done,
            info,
        )
