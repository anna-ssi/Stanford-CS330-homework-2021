'''Test dependencies.'''

import mujoco_py # pylint: disable=unused-import
import gym
import multiworld
import sawyer_action_discretize
multiworld.register_all_envs()

if __name__ == '__main__':
    env = gym.make('SawyerReachXYEnv-v1')
    env = sawyer_action_discretize.SawyerActionDiscretize(
            env, render_every_step=False)
    env.reset()
    for _ in range(10):
        env.step(0)

    print('\n\nDependencies successfully installed!\n\n')
