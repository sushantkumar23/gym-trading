#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Simulate trading environment"

import numpy as np
import gym
from gym import spaces
import logging

from gym_trading.envs.data_src import TrueFXDataSrc
logging.basicConfig(level=logging.INFO)


class TradeEnv(gym.Env):
    """
    Define a simple trading environment.
    The environment defines which actions can be taken at which point and when
    the agent receives which reward.
    """

    def __init__(self, spread=0.00002):
        self.__version__ = "0.1.0"
        logging.info("TradeEnv - Version {}".format(self.__version__))

        # Initialising the environment
        logging.info("spread: {}".format(spread))

        # General variables defining the environment
        self.src = TrueFXDataSrc()
        self.data = self.src.data
        self.n = len(self.data)
        self.spread = spread

        # Define what the agent can do
        self.action_space = spaces.Discrete(3)

    # Step function returns the latest observation. While, state could be
    # defined as stacks of observation. State definition should be taken care
    # of by the agent. An environment just passes the current price of the
    # stock.
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
                the environment. In this case, it is a tuple of the timestamp,
                the close price and the volume traded in that interval.
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
            logging.warn(
                "You are calling 'step()' even though this environment has "
                "already returned done = True. You should always call "
                "'reset()' once you receive 'done = True' -- any further "
                "steps are undefined behavior."
            )

        self.index += 1

        # Get the new Observation
        obs = self._get_observation()

        # Calculate the reward as log returns of that timestep. Log returns are
        # used instead of aritmetic returns since log returns additive.
        step_return = np.log(
            self.data['close'][self.index] / self.data['close'][(self.index-1)]
        )
        commission = self.spread * np.abs(action - self.past_action)
        reward = ((action - 1) * step_return) - commission

        # Update the storage variables
        self.past_action = action

        # Set and assign done
        if self.index >= (self.n - 1):
            self.done = True
        done = self.done

        # Set info
        info = {
            'return': step_return
        }

        return obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (tuple):
            the initial observation of the environment.
        """

        # Initialize the episode
        self.index = 0
        self.done = False
        self.past_action = 0
        return self._get_observation()

    def _get_observation(self):
        """
        Formats the observation in required manner and returns it.
        Returns:
        _______
        observation (tuple):
            observation consists of the three items. A Timestamp,
            np.float64 and an np.int64. The timestamp is the current step, the
            float is the current close price and the int is the volume.
        """
        stock_price = self.data['close'][self.index]
        volume = self.data['volume'][self.index]
        timestamp = self.data.index[self.index]
        obs = (timestamp, stock_price, volume)
        return obs
