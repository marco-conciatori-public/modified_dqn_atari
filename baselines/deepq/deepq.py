import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import time
import datetime

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, LowerBoundReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
from baselines.deepq.defaults import atari


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params, _, _, _ = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        print('loaded model trained with parameters:')
        print(act_params)

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, rep_buffer, lb_buffer, path=None):
        """Save model to a pickle located at `path`"""

        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()

        train_params = atari()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params, train_params, rep_buffer, lb_buffer), f)

    def save(self, path):
        save_variables(path)


def load_play_function(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def load_act(path):
    with open(path, "rb") as f:
        model_data, _, _, replay_buffer, lb_buffer = cloudpickle.load(f)
    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, "packed.zip")
        with open(arc_path, "wb") as f:
            f.write(model_data)

        zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
        load_variables(os.path.join(td, "model"))

    return replay_buffer, lb_buffer


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          train_freq=1,
          replay_batch_size=32,
          lb_batch_size=16,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=10000,
          gamma=1.0,
          load_to_play=False,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    replay_batch_size: int
        max size of a batched sampled from replay buffer for training
    lb_batch_size: int
        max size of a batched sampled from lower bound replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    now = time.time()
    temp_steps = 0
    times = []

    steps_score_data = []

    first_batch_training = True

    sess = get_session()
    set_global_seeds(seed)

    log_dir = '/content/ml-dqn-atari/log'

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    q_values = debug['q_values']

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # # Create the schedule for exploration starting from 1.
    # exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
    #                              initial_p=1.0,
    #                              final_p=exploration_final_eps)

    # Create lower bounds buffer
    lb_buffer = LowerBoundReplayBuffer(buffer_size, gamma)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    lb_used = 0
    replay_counter = 0
    tot_removed_exp = 0
    tot_tot_exp = 1

    num_actions = env.action_space.n

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            if load_to_play:
                # TODO: forse serve modificare ActWrapper.load_act
                load_play_function(load_path)
            else:
                replay_buffer, lb_buffer = load_act(load_path)
                logger.log('Loaded model from {}'.format(load_path))

        tot_time = -time.time()
        memorize_transition_time = 0
        compute_lb_time = 0
        sample_time = 0
        test_time = 0
        q_val_time = 0
        remove_experiences_time = 0
        append_time = 0
        log_time = 0

        # writer = tf.summary.FileWriter(log_dir, graph=sess.graph, flush_secs=60)
        log_time -= time.time()
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)
        log_time += time.time()
        # print('+++++++++++++++ expected log_dir:', log_dir)
        # print('+++++++++++++++ log_dir:', writer.get_logdir())

        got_reward = False
        first_reward = True
        all_negative_counter = 0

        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                # TODO: fare interazione corretta quando load_path non è nullo
                update_param_noise_threshold = 0.
            else:
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            # action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            # random action
            e = 0.001
            action = np.random.choice(num_actions)
            if got_reward:
                actions_q_values = q_values(np.array(obs))[0]
                sum_positive_q_values = 0
                # choice probability
                p = []
                for el in actions_q_values:
                    # scarto le mosse con q-value negativo, a meno che non siano tutti negativi
                    positive_el = max(0, el)
                    p.append(positive_el)
                    sum_positive_q_values += positive_el
                if sum_positive_q_values > 0:
                    normalized_p = [el / sum_positive_q_values for el in p]
                    # action = np.random.choice(num_actions, p=normalized_p)
                    if np.random.uniform(0, 1) > e:
                        action = actions_q_values.argmax()
                else:
                    action = actions_q_values.argmax()
                    # print('got_reward ma sum_positive_q_values <= 0; cioè solo q-values negativi come stima dei valori delle mosse')
                    # print('sum_positive_q_values:', sum_positive_q_values)
                    # print('scelgo in modo deterministico quella migliore (con valore:', max(actions_q_values), ')')
                    all_negative_counter += 1

            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))

            memorize_transition_time -= time.time()
            lb_buffer.memorize_transition(obs, action, rew, new_obs)
            memorize_transition_time += time.time()

            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                if episode_rewards[-1] > 0:
                    got_reward = True
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

                if len(lb_buffer) > 0:
                    test_time -= time.time()
                    removed_exp, tot_exp = lb_buffer.remove_bad_experiences(q_values)
                    test_time += time.time()

                    tot_removed_exp += removed_exp
                    tot_tot_exp += tot_exp

                compute_lb_time -= time.time()
                # lb_buffer.compute_lb()
                lb_buffer.compute_lb(q_values)
                compute_lb_time += time.time()

            if t > learning_starts and t % train_freq == 0:
                if not prioritized_replay:
                    if first_batch_training:
                        print('first_batch_training at episode:', t)
                        first_batch_training = False

                    true_lb_batch_size = 0
                    if len(lb_buffer) > 0:
                        sample_time -= time.time()
                        lb_obses_t, lb_actions, lb_rewards, lb_obses_tp1, lb_dones = lb_buffer.sample(lb_batch_size)
                        sample_time += time.time()

                        q_val_time -= time.time()
                        estimated_rewards = q_values(np.array(lb_obses_t))
                        q_val_time += time.time()
                        true_lb_batch_size = len(lb_obses_t)

                        lb_used += true_lb_batch_size
                    replay_counter += replay_batch_size
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(replay_batch_size - true_lb_batch_size)

                    # print('len(indexes):', len(indexes))
                    # print('++++++++++++++++++++++++++++++++')
                    # print('lb_obses_t shape:', np.shape(lb_obses_t))
                    # print('lb_actions shape:', np.shape(lb_actions))
                    # print('lb_rewards shape:', np.shape(lb_rewards))
                    # print('lb_obses_tp1 shape:', np.shape(lb_obses_tp1))
                    # print('lb_dones shape:', np.shape(lb_dones))
                    # print('obses_t shape:', np.shape(obses_t))
                    # print('actions shape:', np.shape(actions))
                    # print('rewards shape:', np.shape(rewards))
                    # print('obses_tp1 shape:', np.shape(obses_tp1))
                    # print('dones shape:', np.shape(dones))

                    append_time -= time.time()
                    if len(lb_buffer) > 0:
                        obses_t.extend(lb_obses_t)
                        actions.extend(lb_actions)
                        rewards.extend(lb_rewards)
                        obses_tp1.extend(lb_obses_tp1)
                        dones.extend(lb_dones)

                    obses_t = np.array(obses_t)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    obses_tp1 = np.array(obses_tp1)
                    dones = np.array(dones)
                    append_time += time.time()

                    weights = np.ones_like(rewards)
                else:
                    experience = replay_buffer.sample(replay_batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience

                # print('obses_t shape:', obses_t.shape)
                # print('actions shape:', actions.shape)
                # print('rewards shape:', rewards.shape)
                # print('obses_tp1 shape:', obses_tp1.shape)
                # print('dones shape:', dones.shape)
                # print('weights shape:', weights.shape)
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("len(lb_buffer)", len(lb_buffer))

                if not prioritized_replay:
                    if len(lb_buffer) > 0:
                        logger.record_tabular("removed_exp", removed_exp)
                        logger.record_tabular("% removed_exp / tot_exp", 100 * removed_exp / tot_exp)
                    logger.record_tabular("% tot_removed_exp / tot_tot_exp", 100 * tot_removed_exp / tot_tot_exp)
                    if t > learning_starts:
                        logger.record_tabular('% lb usati / replay usati', 100 * lb_used / replay_counter)
                logger.dump_tabular()

                temp_time = now
                now = time.time()
                times.append((now - temp_time, t - temp_steps))
                temp_steps = t
                # print("--- %s seconds ---" % (time.time() - start_time))
                #     print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                    steps_score_data.append((t, saved_mean_reward))

        tot_time += time.time()

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

        step_str = '_steps'
        old_steps = 0
        if load_path is not None:
            index = load_path.find(step_str)
            index += len(step_str)
            char = load_path[index]
            while char != '.':
                if char.isdigit():
                    old_steps = old_steps * 10 + int(char)
                elif char == 'K':
                    old_steps *= 1000
                elif char == 'M':
                    old_steps *= 1000000
                index += 1
                char = load_path[index]

        readable_total_timesteps = total_timesteps + old_steps
        num_m = 0
        num_k = 0
        while readable_total_timesteps % 1000000 == 0:
            num_m += 1
            readable_total_timesteps = readable_total_timesteps // 1000000
        if readable_total_timesteps % 1000 == 0:
            num_k += 1
            readable_total_timesteps = readable_total_timesteps // 1000
        readable_total_timesteps = str(int(readable_total_timesteps))

        for i in range(num_k):
            readable_total_timesteps += 'K'
        for i in range(num_m):
            readable_total_timesteps += 'M'

        if model_saved:
            file_name = env.spec.id + '_newLbTest_rew' + str(saved_mean_reward) + step_str + str(readable_total_timesteps) + '.pkl'
        else:
            file_name = env.spec.id + '_newLbTest_rew' + str(mean_100ep_reward) + step_str + str(readable_total_timesteps) + '.pkl'

        file_path = os.path.join('trained_models', file_name)

        act.save_act(rep_buffer=replay_buffer, lb_buffer=lb_buffer, path=file_path)

        print('times:')
        for ti in times:
            print(ti[0])
        print('steps_times:')
        for ti in times:
            print(ti[1])

        print('steps_score:')
        for el in steps_score_data:
            print(el[0])
        print('score:')
        for el in steps_score_data:
            print(el[1])
        logger.record_tabular("% memorize_transition_time", 100 * memorize_transition_time / tot_time)
        logger.record_tabular("% all negative", 100 * all_negative_counter / total_timesteps)
        logger.record_tabular("% compute_lb_time", 100 * compute_lb_time / tot_time)
        logger.record_tabular("% sample_time", 100 * sample_time / tot_time)
        logger.record_tabular("% test_time", 100 * test_time / tot_time)
        logger.record_tabular("% q_val_time", 100 * q_val_time / tot_time)
        logger.record_tabular("% remove_experiences_time", 100 * remove_experiences_time / tot_time)
        logger.record_tabular("% append_time", 100 * append_time / tot_time)
        logger.record_tabular("% log_time", 100 * log_time / tot_time)
        logger.record_tabular("% tot lb time", 100 * (memorize_transition_time + compute_lb_time + sample_time + q_val_time + test_time + remove_experiences_time + append_time) / tot_time)
        logger.dump_tabular()

        print('total time:', str(datetime.timedelta(seconds=tot_time)))

        writer.close()

    return act
