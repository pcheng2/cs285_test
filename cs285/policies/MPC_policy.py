import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from .base_policy import BasePolicy

#No CEM applied, just through random shotting method


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]

        random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon)

        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(
                obs, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)
            print("predicted_sum_of_rewards_per_model", predicted_sum_of_rewards_per_model.shape)
        # calculate mean_across_ensembles(predicted rewards)
        
        predicted_rewards = np.mean(
            predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N

        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]  # TODO (Q2)
        action_to_take = best_action_sequence[0]  # TODO (Q2)
        print("action_to_take", action_to_take)
        return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        N = candidate_action_sequences.shape[0]
        sum_of_rewards = np.zeros(N)

        assert candidate_action_sequences.shape[1] == self.horizon

        obs = np.tile(obs, (N, 1)) #repeat axis=0 for N times, axis=1 for 1 time(not repeat at all)

        #array([[0, 1, 2],
        #       [3, 4, 5]])
        #y = np.tile(x,(2,3))
        #array([[0, 1, 2, 0, 1, 2, 0, 1, 2],
        #       [3, 4, 5, 3, 4, 5, 3, 4, 5],
        #
        #       [0, 1, 2, 0, 1, 2, 0, 1, 2],
        #       [3, 4, 5, 3, 4, 5, 3, 4, 5]])

        #In here, horizon is a fixed number
        for _ in range(self.horizon):
            actions = candidate_action_sequences[:,_,:]
            rewards, done = self.env.get_reward(obs,actions)
            sum_of_rewards += rewards
            obs = model.get_prediction(obs, actions, self.data_statistics)

        return sum_of_rewards
