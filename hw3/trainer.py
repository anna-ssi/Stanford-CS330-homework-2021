"""Main training loop used for training both sawyer_main and bit flipping.

Modify the code in the sections marked TODO."""

import numpy as np
import torch
from torch import nn
from torch import optim

import replay_buffer as experience_buffer
import q_network
import run_episode
import utils
from utils import HERType

def update_replay_buffer(
    replay_buffer,
    episode_experience,
    her_type=HERType.NO_HINDSIGHT,
    env_reward_function=None, # pylint: disable=unused-argument
    num_relabeled=4, # pylint: disable=unused-argument
):
    """Adds past experience to the replay buffer. Training is done with
    episodes from the replay buffer. When HER is used, relabeled
    experiences are also added to the replay buffer.

    Args:
        replay_buffer (ReplayBuffer): replay buffer to store experience
        episode_experience (list): list containing the transitions
            (state, action, reward, next_state, goal_state)
        HER (HERType): type of hindsight experience replay to use
        env_reward_function ((ndarray, ndarray) -> float):
            reward function for relabelling transitions
        num_relabeled (int): number of relabeled transition per transition
    """

    for timestep in range(len(episode_experience)):

        # copy experience from episode_experience to replay_buffer
        state, action, reward, next_state, goal = episode_experience[timestep]
        # use replay_buffer.add
        replay_buffer.add(np.append(state, goal),
                          action,
                          reward,
                          np.append(next_state, goal))

        # ======================== TODO modify code ========================

        if her_type == HERType.FINAL:
            # relabel episode based on final state in episode
            pass

            # get final goal

            # compute new reward

            # add to buffer

        elif her_type == HERType.FUTURE:
            # future: relabel episode based on randomly sampled future state.
            # At each timestep t, relabel the goal with a randomly selected
            # timestep between t and the end of the episode
            pass

            # for every transition, add num_relabeled transitions to the buffer

            # get random future goal

            # compute new reward

            # add to replay buffer

        elif her_type == HERType.RANDOM:
            # random: relabel episode based on a random state from the episode
            pass

            # for every transition, add num_relabeled transitions to the buffer

            # get random goal

            # compute new reward

            # add to replay buffer

        # ========================      END TODO       ========================

def train(
    env,
    input_dim,
    action_dim,
    num_epochs,
    writer,
    her_type=HERType.NO_HINDSIGHT,
    env_reward_function=None,
    num_relabeled=4,
    buffer_size=1e6,
    num_episodes=16,
    steps_per_episode=50,
    gamma=0.98,
    opt_steps=40,
    batch_size=128,
    log_interval=5,
):
    """Main loop for training DQN on the sawyer environment. The DQN is
    trained for num_epochs. In each epoch, the agent runs in the environment
    num_episodes number of times. The Q-target and Q-policy networks are
    updated at the end of each epoch. Within one episode, Q-policy attempts
    to solve the environment and is limited to the same number as steps as the
    size of the environment.

    Args:
        env (gym object): main environment to sample transitions.
        input_dim (int): input size for the Q-network.
        action_dim (int): action space for the environment,
            output dim for Q-network.
        num_epochs (int): number of epochs to train DQN for
        her_type (HERType): Specify version of hindsight experience replay,
            defaults to None
        num_relabeled (int): number of relabelled trajectories per
            collected trajectory when using HER
        buffer_size (int): number of recent experiences to store in the
            replay buffer
        num_episodes (int): number of goals attempted per epoch
        steps_per_episode (int): number of steps_per_episode
        gamma (float): discount factor for RL
        opt_steps (int): number of gradient steps per epoch
        batch_size (int): number of transitions sampled from the
            replay buffer per optimization step
        writer (tensorboard.SummaryWriter): tensorboard event logger
        log_interval (int): frequency for recording tensorboard events
    """

    # create replay buffer
    replay_buffer = experience_buffer.Buffer(buffer_size, batch_size)

    # set up Q-policy (model) and Q-target (target_model)
    model = q_network.QNetwork(input_dim, action_dim)
    target_model = q_network.QNetwork(input_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # start by making Q-target and Q-policy the same
    utils.update_target(model, target_model)

    # Run for a fixed number of epochs
    for epoch_idx in range(num_epochs):
        # total reward for the epoch
        total_reward = 0.0
        # record success rate for each episode of the epoch
        successes = []
        # loss at the end of each epoch
        losses = []

        for _ in range(num_episodes):
            # collect data in the environment
            episode_experience, ep_reward, succeeded = run_episode.run_episode(
                env, model, steps_per_episode)
            # track eval metrics in the environment
            total_reward += ep_reward
            successes.append(succeeded)
            # add to the replay buffer; use specified HER policy
            update_replay_buffer(
                replay_buffer,
                episode_experience,
                her_type=her_type,
                env_reward_function=env_reward_function,
                num_relabeled=num_relabeled
            )

        # optimize the Q-policy network
        for _ in range(opt_steps):
            # sample from the replay buffer
            state, action, reward, next_state = replay_buffer.sample()
            state = torch.from_numpy(state.astype(np.float32))
            action = torch.from_numpy(action)
            reward = torch.from_numpy(reward.astype(np.float32))
            next_state = torch.from_numpy(next_state.astype(np.float32))

            optimizer.zero_grad()
            # forward pass through target network
            target_q_vals = target_model(next_state).detach()
            # calculate target reward
            q_loss_target = torch.clip(
                reward + gamma * torch.max(target_q_vals, axis=-1).values,
                -1.0 / (1 - gamma),
                0)
            # calculate predictions and loss
            model_predict = model(state)
            model_action_taken = torch.reshape(action, [-1])
            action_one_hot = nn.functional.one_hot(
                model_action_taken, action_dim)
            q_val = torch.sum(model_predict * action_one_hot, axis=1)
            criterion = nn.MSELoss()
            loss = criterion(q_val, q_loss_target)
            losses.append(loss.detach().numpy())

            loss.backward()
            optimizer.step()

        # update target model by copying Q-policy to Q-target
        utils.update_target(model, target_model)

        if epoch_idx % log_interval == 0:
            print(
                f"Epoch: {epoch_idx} Cumulative reward: {total_reward} Success rate: {np.mean(successes)} Mean loss: {np.mean(losses)}" # pylint: disable=line-too-long
            )
            writer.add_scalar(
                "eval_metrics/total_reward", total_reward, epoch_idx)
            writer.add_scalar(
                "eval_metrics/success_rate", np.mean(successes), epoch_idx)
            writer.add_scalar(
                "train_metrics/td_loss", np.mean(losses), epoch_idx)
