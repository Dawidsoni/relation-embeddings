from typing import List
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
        phases: List[float] = gin.REQUIRED,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.phases = phases

    def _get_current_phase_step(self, global_step):
        if len(self.phases) == 0 or global_step < self.phases[0]:
            return global_step
        step = None
        for index, phase in enumerate(self.phases):
            if global_step - phase >= 0:
                step = global_step - phase
        return step

    def __call__(self, global_step):
        step = self._get_current_phase_step(global_step)
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
