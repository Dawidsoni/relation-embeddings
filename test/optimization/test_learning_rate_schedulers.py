import tensorflow as tf

from optimization.learning_rate_schedulers import PiecewiseLinearDecayScheduler


class TestLearningRateSchedulers(tf.test.TestCase):

    def test_decay_rate_one(self):
        scheduler = PiecewiseLinearDecayScheduler(
            initial_learning_rate=0.1, decay_steps=100, decay_rate=1.0, warmup_steps=0,
        )
        self.assertAllClose(0.1, scheduler(0))
        self.assertAllClose(0.1, scheduler(100))
        self.assertAllClose(0.1, scheduler(200))

    def test_fold_points(self):
        scheduler = PiecewiseLinearDecayScheduler(
            initial_learning_rate=0.3, decay_steps=100, decay_rate=0.1, warmup_steps=0,
        )
        self.assertAllClose(0.3, scheduler(0))
        self.assertAllClose(0.03, scheduler(100))
        self.assertAllClose(0.003, scheduler(200))

    def test_points_between_folds(self):
        scheduler = PiecewiseLinearDecayScheduler(
            initial_learning_rate=0.1, decay_steps=100, decay_rate=0.1, warmup_steps=0,
        )
        self.assertAllClose(0.08, scheduler(20), atol=5e-3)
        self.assertAllClose(0.05, scheduler(50), atol=5e-3)
        self.assertAllClose(0.008, scheduler(120), atol=5e-4)
        self.assertAllClose(0.0002, scheduler(280), atol=1e-4)

    def test_warmup_steps(self):
        scheduler = PiecewiseLinearDecayScheduler(
            initial_learning_rate=0.1, decay_steps=100, decay_rate=0.1, warmup_steps=250,
        )
        self.assertAllClose(0.02, scheduler(49))
        self.assertAllClose(0.05, scheduler(124))
        self.assertAllClose(0.1, scheduler(249))
        self.assertAllClose(0.1, scheduler(250))
        self.assertAllClose(0.08, scheduler(270), atol=5e-3)
        self.assertAllClose(0.01, scheduler(350))

    def test_optimizer_compatible(self):
        scheduler = PiecewiseLinearDecayScheduler(
            initial_learning_rate=0.1, decay_steps=100, decay_rate=0.1, warmup_steps=0,
        )
        optimizer = tf.keras.optimizers.Adam(scheduler)
        model_variable = tf.Variable(initial_value=1.0)
        optimizer.apply_gradients(zip([1.0], [model_variable]))
        optimizer.apply_gradients(zip([1.0], [model_variable]))
        self.assertAllClose(0.8009, model_variable)


if __name__ == '__main__':
    tf.test.main()
