"""Convert discrete action into continuous action for Sawyer."""

import gym
import numpy as np


class SawyerActionDiscretize(gym.Wrapper):
    '''
    Environment wrapper to discretize sawyer's action space.
    Simplifies the observation space to (x, y) position of the arm.
    '''
    def __init__(self, env, done_threshold=-0.03, render_every_step=False):
        super().__init__(env)
        self._done_threshold = done_threshold
        self._render_every_step = render_every_step

    def reset(self):
        """
        Reset the environment, and independently return the
        state and the goal state.

        Returns:
            new_state (ndarray): numpy array with the arm location
            goal_state (ndarray): numpy array with the goal location
        """
        # reset base environment
        reset_state = self.env.reset()
        # split into observation and goal
        return reset_state["observation"][:2], reset_state["desired_goal"][:2]

    def step(self, action):
        """Takes an action in the sawyer environments.

        Passes the discrete action selected by the Q-network to the
        sawyer Arm. The function returns the next state, the reward,
        and whether the environment was solved in info.

        Args:
            action (int): discrete action in [0,  NUM_ACT-1]

        Returns:
            next_state (ndarray): numpy array with arm location
            reward (ndarray): reward returned by the sawyer environment
            done (bool): boolean indicating whether the environment was solved
            info (dict): contains 'successful_this_state' and other info"""

        # maps actions selected by Q-network to Sawyer arm actions
        action_dic = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        # map discrete action to continuous action
        action_sawyer = np.array(action_dic[action], dtype=np.float32)
        # take the action
        ob, reward, done, info = self.env.step(action_sawyer)
        # if rendering is turned on, render the environment
        if self._render_every_step:
            self.env.render()
        # check if we're "close enough" to declare done
        info["successful_this_state"] = reward > self._done_threshold

        # pull the observed state off
        next_state = ob["observation"][0:2]

        return next_state, reward, done, info
