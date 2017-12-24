
class GymEnvWrapper(object):
    def __init__(self, env, repeat_actions=1):
        self.env = env
        self.repeat_actions = repeat_actions
        self.loss_reward = 1.0

    def __getattr__(self, attr):
        org_attr = self.env.__getattribute__(attr)
        return org_attr

    def step(self, action, is_training=True):
        start_lives = self.lives
        total_reward = 0
        for i in range(self.repeat_actions):
            observation, reward, terminal, info = self.env.step(action)
            total_reward += reward
        if is_training:
            total_reward += (self.lives - start_lives) * self.loss_reward
        return observation, total_reward, terminal, info

    @property
    def lives(self):
        return self.env.env.ale.lives()
