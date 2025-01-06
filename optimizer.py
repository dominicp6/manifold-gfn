# TODO: figure out why the z_dim can be larger than 1
# TODO: match lr_z_mult with that of original code

class OptimizerConfig:
    def __init__(self, 
                 loss='trajectorybalance', 
                 lr=0.0001,
                 lr_z_mult = 3, 
                 lr_decay_period=1000000, 
                 lr_decay_gamma=0.5, 
                 z_dim=16,
                 method='adam', 
                 early_stopping=0.0, 
                 ema_alpha=0.5, 
                 adam_beta1=0.9, 
                 adam_beta2=0.999, 
                 sgd_momentum=0.9,  
                 train_to_sample_ratio=1, 
                 n_train_steps=5000, 
                 bootstrap_tau=0.0, 
                 clip_grad_norm=0.0,
                 clip_value=1e-5,
                 steps_per_batch=3,
                 gradient_clipping = True,):
        self.z_dim = z_dim
        self.loss = loss
        self.lr = lr
        self.lr_z_mult = lr_z_mult
        self.lr_decay_period = lr_decay_period
        self.lr_decay_gamma = lr_decay_gamma
        self.method = method
        self.early_stopping = early_stopping
        self.ema_alpha = ema_alpha
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.sgd_momentum = sgd_momentum
        self.train_to_sample_ratio = train_to_sample_ratio
        self.n_train_steps = n_train_steps
        self.bootstrap_tau = bootstrap_tau
        self.clip_grad_norm = clip_grad_norm
        self.clip_value = clip_value
        self.steps_per_batch = steps_per_batch
        self.gradient_clipping = gradient_clipping