#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Simulate trading environment"

import logging

import numpy as np
import gym
from gym import spaces, logger

from gym_trading.envs.data_src import TrueFXDataSrc


class Series(gym.Space):

    def __init__(self, data, window):
        self.shape = (window,)

        # For multiple stocks
        # self.series = np.vstack([np.array([data[key][i-window:i]for i in
        # range(window, (len(data[key]) + 1))]) for key in list(data.keys())])

    def sample(self):
        return self.series[np.random.randint(len(self.series))]


class TradeEnv(gym.Env):
    """
    Define a simple trading environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, spread=0.08):
        self.__version__ = "0.1.0"
        logger.info("Tradenv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.src = TrueFXDataSrc()
        self.data = self.src.data
        self.series = self.src.series
        self.n = len(self.series)
        self.spread = spread

        # Define what the agent can do
        self.action_space = spaces.Discrete(3)
        self.v = 0

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (tuple) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        # Check if the action is valid
        assert self.action_space.contains(action)

        # Check if the env is already done
        if self.done:
            logger.warn(
                "You are calling 'step()' even though this environment has "
                "already returned done = True. You should always call "
                "'reset()' once you receive 'done = True' -- any further "
                "steps are undefined behavior."
            )

        self.index += 1

        # Get the new Observation
        obs = self._get_observation()

        # Calculate the reward
        diff = self.series[self.index] - self.series[self.index - 1]
        commission = self.spread * np.abs(action - self.past_action)
        self.v_delta = ((action - 1) * diff) - commission
        # Rewards are calculated as log returns since log returns are additive
        reward = np.log((self.v + self.v_delta)/self.v)

        # Update the storage variables
        self.v += self.v_delta
        self.past_action = action

        # Set and assign done
        if self.index >= (self.n - 1):
            self.done = True
        done = self.done

        # Set info
        info = {}

        return obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (tuple): the initial observation of the space.
        """

        # Initialize the episode
        self.index = 0
        self.done = False
        self.past_action = 0
        return self._get_observation()

    def _get_observation(self):
        """Get the observation."""
        stock_price = self.series[self.index]
        timestamp = self.series.index[self.index]
        obs = (timestamp, stock_price)
        return obs
