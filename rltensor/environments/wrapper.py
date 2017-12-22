
class RepeatEnvWrapper(object):
    def __init__(self, env, repeat_actions=1):
        self.env = env
        self.repeat_actions = repeat_actions

    def __getattr__(self, attr):
        org_attr = self.env.__getattribute__(attr)
        return org_attr

    def step(self, action):
        total_reward = 0
        for i in range(self.repeat_actions):
            observation, reward, terminal, info = self.env.step(action)
            total_reward += reward
        return observation, total_reward, terminal, info
