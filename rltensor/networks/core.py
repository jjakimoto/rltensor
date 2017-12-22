class BaseNetwork(object):
    def __init__(self, model_params, scope_name, *args, **kwargs):
        self.model_params = model_params
        self.scope_name = scope_name
        self.reg_loss = 0.
        self.reuse = False

    def __call__(self, x, training=None):
        raise NotImplementedError()

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_reg_loss(self):
        return self.reg_loss

    def get_training(self):
        return self.training
