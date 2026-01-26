import lightning as L
from lightning.pytorch.callbacks import Callback

class RolloutValidationCallback(Callback):
    def __init__(self, every_n_epochs=2):
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, l_module):
        epoch = trainer.current_epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        rewards = l_module.run_validation_rollout()
        self.log(
            "val_episode_return",
            float(rewards),
            prog_bar=True
        )
