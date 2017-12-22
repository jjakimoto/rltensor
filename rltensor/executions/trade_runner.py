from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
from six.moves import xrange
import numpy as np
from .runner import Runner


class TradeRunner(Runner):
    def __init__(self, agent, env, env_name="trading",
                 tensorboard_dir="./logs", scalar_summary_tags=None,
                 histogram_summary_tags=None, load_file_path=None,
                 *args, **kwargs):
        if scalar_summary_tags is None:
            scalar_summary_tags = [
                'average.loss',
                'average.returns',
                'drawdown',
                'cumulative_returns',
                'episode.final_value',
                'episode.max_returns',
                'episode.min_returns',
                'episode.avg_returns',
                'episode.maximum_drawdowns',
                'episode.sharp_ratio',
                'training.learning_rate',
                'training.num_step_per_sec',
                'training.time',
                'test.cumulative_returns',
                'test.drawdowns',
                'test.final_value',
                'test.maximum_drawdowns',
                'test.sharp_ratio'
            ]
        self.scalar_summary_tags = scalar_summary_tags

        if histogram_summary_tags is None:
            histogram_summary_tags = ['episode.returns', 'test.returns']
            for i in range(env.action_dim):
                histogram_summary_tags.append("episode.action_{}".format(i))
                histogram_summary_tags.append("test.action_{}".format(i))

        super(TradeRunner, self).__init__(agent=agent, env=env,
                                          env_name=env_name,
                                          tensorboard_dir=tensorboard_dir,
                                          scalar_summary_tags=scalar_summary_tags,
                                          histogram_summary_tags=histogram_summary_tags,
                                          load_file_path=load_file_path,
                                          *args, **kwargs)

    def fit(self, t_max, num_max_start_steps=0,
            save_file_path=None,
            overwrite=True,
            log_freq=1000,
            avg_length=1000,
            init_pv=100.,
            *args, **kwargs):
        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
        self._reset(self.agent, self.env)

        self.sp_list = []
        self.mmd_list = []
        self.pv_list = []
        self.accumulated_pvs_list = []
        self.draw_downs_list = []
        # accumulate results
        total_reward = deque(maxlen=avg_length)
        total_loss = deque(maxlen=avg_length)
        total_q_val = deque(maxlen=avg_length)
        total_q_max_val = deque(maxlen=avg_length)
        ep_rewards = []
        ep_losses = []
        ep_q_vals = []
        ep_actions = []
        ep_errors = []
        step = self.agent.global_step
        _st = self.st

        # Determine if it has to be randomly initialized
        init_flag = True
        # Start from the middle of training
        # t_max = t_max - step
        _env = self.env

        episode_start_time = time.time()

        self.agent.reset()
        state = self.environment.reset()
        accumulated_pv = init_pv
        peak_pv = init_pv
        self.accumulated_pvs = [accumulated_pv]
        draw_downs = [0.]
        returns = [0.]
        self.episode_timestep = 0
        try:
            for t in tqdm(xrange(t_max)):
                if init_flag:
                    init_flag = False
                    if num_max_start_steps == 0:
                        num_random_start_steps = 0
                    else:
                        num_random_start_steps =\
                            np.random.randint(num_max_start_steps)
                    for _ in xrange(num_random_start_steps):
                        action = _env.action_space.sample()
                        observation, reward, terminal, info =\
                            _env.step(action)
                        if terminal:
                            self._reset(self.agent, _env)
                        self.agent.observe(observation, action,
                                           reward, terminal,
                                           training=False, is_store=False)
                # Update step
                self.agent.update_step()
                step = self.agent.global_step
                # 1. predict
                state = self.agent.get_recent_state()
                action = self.agent.predict(state)
                # 2. act
                observation, reward, terminal, info = _env.step(action)
                # 3. store data and train network
                if step < self.agent.t_learn_start:
                    response = self.agent.observe(observation, action, reward,
                                                  terminal, training=False,
                                                  is_store=True)
                    if terminal:
                        self._reset(self.agent, _env)
                else:
                    response = self.agent.observe(observation, action, reward,
                                                  terminal, training=True,
                                                  is_store=True)

                    loss, is_update = response
                    step = self.agent.global_step
                    # update statistics
                    total_reward.append(reward)
                    total_loss.append(loss)
                    total_q_val.append(np.mean(q))
                    total_q_max_val.append(np.mean(q_max))
                    ep_actions.append(action)
                    ep_errors.append(error)
                    ep_rewards.append(reward)
                    ep_losses.append(loss)
                    ep_q_vals.append(np.mean(q))
                    # Write summary
                    if log_freq is not None and step % log_freq == 0:
                        num_per_sec = log_freq / (time.time() - _st)
                        _st = time.time()
                        epsilon = self.agent.epsilon
                        learning_rate = self.agent.learning_rate
                        avg_r = np.mean(total_reward)
                        avg_loss = np.mean(total_loss)
                        avg_q_val = np.mean(total_q_val)
                        avg_q_max_val = np.mean(total_q_max_val)
                        tag_dict = {'episode.num_of_game': self.num_ep,
                                    'average.reward': avg_r,
                                    'average.loss': avg_loss,
                                    'average.q': avg_q_val,
                                    'average.q_max': avg_q_max_val,
                                    'training.epsilon': epsilon,
                                    'training.learning_rate': learning_rate,
                                    'training.num_step_per_sec': num_per_sec,
                                    'training.time': time.time() - self.st}
                        self._inject_summary(tag_dict, step)
                    if terminal:
                        try:
                            cum_ep_reward = np.sum(ep_rewards)
                            max_ep_reward = np.max(ep_rewards)
                            min_ep_reward = np.min(ep_rewards)
                            avg_ep_reward = np.mean(ep_rewards)
                        except:
                            cum_ep_reward = 0
                            max_ep_reward = 0
                            min_ep_reward = 0
                            avg_ep_reward = 0

                        tag_dict = {'episode.cumulative_reward': cum_ep_reward,
                                    'episode.max_reward': max_ep_reward,
                                    'episode.min_reward': min_ep_reward,
                                    'episode.avg_reward': avg_ep_reward,
                                    'episode.rewards': ep_rewards,
                                    'episode.actions': ep_actions,
                                    'episode.errors': ep_errors}
                        self._inject_summary(tag_dict, self.num_ep)
                        # Reset stored current states
                        self._reset(self.agent, _env)
                        ep_rewards = []
                        ep_losses = []
                        ep_q_vals = []
                        ep_actions = []
                        ep_errors = []
                        self.num_ep += 1
                        init_flag = True
        except KeyboardInterrupt:
            pass
        # Update parameters before finishing
        self.agent.save_params(save_file_path, True)

                return_, accumulated_pv, peak_pv, draw_down = calc_stats(log_return, accumulated_pv, peak_pv)
                returns.append(return_)
                self.accumulated_pvs.append(accumulated_pv)
                draw_downs.append(draw_down)

                if terminal or self.agent.should_stop():  # TODO: should_stop also termina?
                    break

            time_passed = time.time() - episode_start_time

            self.pv_list.append(self.accumulated_pvs[-1])
            self.mmd_list.append(np.max(draw_downs))
            self.sp_list.append(calc_sharp_ratio(returns))
            self.accumulated_pvs_list.append(self.accumulated_pvs)
            self.draw_downs_list.append(draw_downs)

            self.episode_timesteps.append(self.episode_timestep)
            self.episode_times.append(time_passed)

            self.episode += 1

            if episode_finished and not episode_finished(self) or \
                    (episodes is not None and self.agent.episode >= episodes) or \
                    (timesteps is not None and self.agent.timestep >= timesteps) or \
                    self.agent.should_stop():
                # agent.episode / agent.timestep are globally updated
                break

        self.agent.close()
        self.environment.close()
