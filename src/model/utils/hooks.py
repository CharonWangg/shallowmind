from pytorch_lightning.callbacks import Callback


class OptimizerResumeHook(Callback):
    def on_train_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            optimizer.param_groups[0]["capturable"] = True