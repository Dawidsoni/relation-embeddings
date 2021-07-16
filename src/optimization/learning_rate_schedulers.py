import tensorflow as tf
import gin.tf


@gin.configurable
class PiecewiseLinearDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        initial_learning_rate: float = gin.REQUIRED,
        decay_steps: int = gin.REQUIRED,
        decay_rate: float = gin.REQUIRED,
        warmup_steps: float = gin.REQUIRED,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_learning_rate * (step + 1) / self.warmup_steps
        steps_since_warmup = step - self.warmup_steps
        last_fold_point = steps_since_warmup // self.decay_steps
        last_fold_learning_rate = self.initial_learning_rate * (self.decay_rate ** last_fold_point)
        next_fold_learning_rate = self.initial_learning_rate * (self.decay_rate ** (last_fold_point + 1))
        linear_rate = (steps_since_warmup % self.decay_steps) / self.decay_steps
        return last_fold_learning_rate - (last_fold_learning_rate - next_fold_learning_rate) * linear_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "warmup_steps": self.warmup_steps,
        }
