import lightning as L

class RolloutValidationCallback(L.Callback):
    def __init__(self, every_n_epochs=2):
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, l_module):
        epoch = trainer.current_epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        metric = l_module.run_validation_rollout()

        l_module.log(
            "val/episode_return",
            metric,
            prog_bar=True,
            sync_dist=True,
        )
