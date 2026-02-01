import lightning as L
from lightning.pytorch.callbacks import Callback

class RolloutValidationCallback(Callback):
    def __init__(self, every_n_epochs=2):
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, l_module):
        epoch = trainer.current_epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        rewards = l_module.run_validation_rollout(train=True)
        self.log(
            "val_episode_return",
            float(rewards),
            prog_bar=True
        )

class RolloutTestCallback(Callback):
    def __init__(self, n_episodes=2, compute_rewards=False, sample=True):
        self.n_episodes = n_episodes
        self.compute_rewards = compute_rewards
        self.sample = sample
    
    def on_test_epoch_start(self, trainer, l_module):
        for i in range(self.n_episodes):
            rewards = l_module.run_validation_rollout(
                train=self.compute_rewards,
                sample=self.sample
            )
            print(f"Episode {i+1} rewards: {float(rewards):.2f}.")     
