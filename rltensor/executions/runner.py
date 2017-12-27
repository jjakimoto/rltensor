from tqdm import tqdm
import tensorflow as tf
import time
from collections import deque
from six.moves import xrange
from gym import wrappers
import numpy as np


class RunnerMixin(object):
    def _build_tags(self, scalar_summary_tags=None,
                    histogram_summary_tags=None):
        if scalar_summary_tags is None:
            scalar_summary_tags = [
                'average.reward', 'average.loss', 'average.q',
                'average.q_max',
                'episode.cumulative_reward', 'episode.max_reward',
                'episode.min_reward', 'episode.avg_reward',
                'episode.num_of_game', 'training.epsilon',
                'training.learning_rate',
                'training.num_step_per_sec', 'training.time']
        self.scalar_summary_tags = scalar_summary_tags

        if histogram_summary_tags is None:
            histogram_summary_tags = ['episode.rewards',
                                      'episode.actions',
                                      'episode.errors',
                                      'episode_test.cumulative_rewards',
                                      'episode_test.max_rewards',
                                      'episode_test.min_rewards',
                                      'episode_test.avg_rewards']
        self.histogram_summary_tags = histogram_summary_tags

    def fit(self, t_max, num_max_start_steps=0,
            save_file_path=None,
            overwrite=True,
            render_freq=None,
            log_freq=1000,
            avg_length=1000):
        # Save Model
        self.save_params(save_file_path, overwrite)
        # Record Viodeo
        _env = self.env
        self._reset(_env)
        # initialize target netwoork
        self.init_update()
        # accumulate results
        self._build_recorders(avg_length)
        step = self.global_step
        # Determine if it has to be randomly initialized
        self.init_flag = True
        # Start from the middle of training
        t_max = t_max - step
        self.st = time.time()
        self.record_st = self.st
        try:
            for t in tqdm(xrange(t_max)):
                if self.init_flag:
                    self.init_flag = False
                    if num_max_start_steps == 0:
                        num_random_start_steps = 0
                    else:
                        num_random_start_steps =\
                            np.random.randint(num_max_start_steps)
                    for _ in xrange(num_random_start_steps):
                        action = _env.action_space.sample()
                        observation, reward, terminal, info =\
                            _env.step(action, is_training=True)
                        if terminal:
                            self._reset(_env)
                        self.observe(observation, action,
                                     reward, terminal, info,
                                     training=False, is_store=False)
                # Update step
                self.update_step()
                step = self.global_step
                # 1. predict
                state = self.get_recent_state()
                action = self.predict(state)
                # 2. act
                observation, reward, terminal, info =\
                    _env.step(action, is_training=True)
                self._update_status(observation, reward, terminal, info)
                # 3. store data and train network
                if step < self.t_learn_start:
                    response = self.observe(observation, action, reward,
                                            terminal, info, training=False,
                                            is_store=True)
                else:
                    response = self.observe(observation, action, reward,
                                            terminal, info, training=True,
                                            is_store=True)
                    self._record(observation, reward, terminal, info,
                                 action, response, log_freq)
                # Visualize reuslts
                if render_freq is not None:
                    if step % render_freq == 0:
                        _env.render()
                # Reset environment
                if terminal:
                    self._reset(_env)
                    self.update_episode()
        except KeyboardInterrupt:
            pass
        # Update parameters before finishing
        self.save_params(save_file_path, True)

    def play(self, num_episode=1, ep=0.05, overwrite=True,
             load_file_path=None, save_video_path=None, render_freq=None):
        if load_file_path is not None:
            tf.global_variables_initializer().run(session=self.sess)
            self.load_params(load_file_path)
        # Record Viodeo
        _env = self.env
        self._reset(_env)
        if save_video_path is not None:
            _env = wrappers.Monitor(_env,
                                    save_video_path,
                                    force=overwrite)

        self._build_recorders_play()
        for num_ep in range(1, num_episode + 1):
            # initialize enviroment
            self._reset(_env)
            terminal = False
            step = 1
            while not terminal:
                # 1. predict
                state = self.get_recent_state()
                action = self.predict(state, ep)
                # 2. act
                observation, reward, terminal, info =\
                    _env.step(action, is_training=False)
                self.observe(observation, action, reward, terminal,
                             info, training=False, is_store=False)
                if render_freq is not None:
                    if step % render_freq == 0:
                        _env.render()
                # Record results
                self._record_play(observation, reward, terminal, info)
                step += 1

    def _build_summaries(self):
        self._build_tags()
        self.writer = tf.summary.FileWriter(
            self.tensorboard_dir, self.sess.graph)
        self.summary_placeholders = {}
        self.summary_ops = {}
        for tag in self.scalar_summary_tags:
            self.summary_placeholders[tag] =\
                tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag] =\
                tf.summary.scalar("%s/%s" % (self.env_name, tag),
                                  self.summary_placeholders[tag])

        for tag in self.histogram_summary_tags:
            self.summary_placeholders[tag] =\
                tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag] = tf.summary.histogram(
                tag,
                self.summary_placeholders[tag])

    def _inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run(
            [self.summary_ops[tag] for tag in tag_dict.keys()],
            {self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)

    def _reset(self, env):
        self.memory.reset()
        observation = env.reset()
        self.observe(observation, None, 0, False,
                     None, training=False, is_store=False)
        return observation

    def _record(self, observation, reward, terminal, info,
                action, response, log_freq):
        q, q_max, loss, error, is_update = response
        step = self.global_step
        # update statistics
        self.total_reward.append(reward)
        self.total_loss.append(loss)
        self.total_q_val.append(np.mean(q))
        self.total_q_max_val.append(np.mean(q_max))
        self.ep_actions.append(action)
        self.ep_errors.append(error)
        self.ep_rewards.append(reward)
        self.ep_losses.append(loss)
        self.ep_q_vals.append(np.mean(q))
        # Write summary
        if log_freq is not None and step % log_freq == 0:
            num_per_sec = log_freq / (time.time() - self.record_st)
            self.record_st = time.time()
            epsilon = self.epsilon
            learning_rate = self.learning_rate
            avg_r = np.mean(self.total_reward)
            avg_loss = np.mean(self.total_loss)
            avg_q_val = np.mean(self.total_q_val)
            avg_q_max_val = np.mean(self.total_q_max_val)
            tag_dict = {'episode.num_of_game': self.num_episode,
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
                cum_ep_reward = np.sum(self.ep_rewards)
                max_ep_reward = np.max(self.ep_rewards)
                min_ep_reward = np.min(self.ep_rewards)
                avg_ep_reward = np.mean(self.ep_rewards)
            except:
                cum_ep_reward = 0
                max_ep_reward = 0
                min_ep_reward = 0
                avg_ep_reward = 0

            tag_dict = {'episode.cumulative_reward': cum_ep_reward,
                        'episode.max_reward': max_ep_reward,
                        'episode.min_reward': min_ep_reward,
                        'episode.avg_reward': avg_ep_reward,
                        'episode.rewards': self.ep_rewards,
                        'episode.actions': self.ep_actions,
                        'episode.errors': self.ep_errors}
            self._inject_summary(tag_dict, self.num_episode)
            # Reset stored current states
            self.ep_rewards = []
            self.ep_losses = []
            self.ep_q_vals = []
            self.ep_actions = []
            self.ep_errors = []
            self.init_flag = True

    def _record_play(self, observation, reward, terminal, info):
        # accumulate results
        self.ep_rewards.append(reward)
        if terminal:
            try:
                cum_ep_reward = np.sum(self.ep_rewards)
                max_ep_reward = np.max(self.ep_rewards)
                min_ep_reward = np.min(self.ep_rewards)
                avg_ep_reward = np.mean(self.ep_rewards)
            except:
                cum_ep_reward = 0
                max_ep_reward = 0
                min_ep_reward = 0
                avg_ep_reward = 0
            self.cum_ep_rewards.append(cum_ep_reward)
            self.max_ep_rewards.append(max_ep_reward)
            self.min_ep_rewards.append(min_ep_reward)
            self.avg_ep_rewards.append(avg_ep_reward)

            tag_dict = {'episode_test.cumulative_rewards': cum_ep_reward,
                        'episode_test.max_rewards': max_ep_reward,
                        'episode_test.min_rewards': min_ep_reward,
                        'episode_test.avg_rewards': avg_ep_reward}
            self._inject_summary(tag_dict, self.num_episode)

    def _build_recorders(self, avg_length):
        self.total_reward = deque(maxlen=avg_length)
        self.total_loss = deque(maxlen=avg_length)
        self.total_q_val = deque(maxlen=avg_length)
        self.total_q_max_val = deque(maxlen=avg_length)
        self.ep_rewards = []
        self.ep_losses = []
        self.ep_q_vals = []
        self.ep_actions = []
        self.ep_errors = []

    def _build_recorders_play(self, avg_length=None):
        self.ep_rewards = []
        self.cum_ep_rewards = []
        self.max_ep_rewards = []
        self.min_ep_rewards = []
        self.avg_ep_rewards = []

    def _update_status(self, observation, reward, terminal, info):
        pass
