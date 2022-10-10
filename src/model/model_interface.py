from copy import deepcopy
import torch
import pytorch_lightning as pl
from shallowmind.src.model.builder import build_arch, build_optimizer, build_scheduler, build_metric
from shallowmind.src.data.pipeline import Compose


class ModelInterface(pl.LightningModule):
    def __init__(self, model, optimization):
        super().__init__()
        self.save_hyperparameters()
        self.model = deepcopy(model)
        self.optimization = deepcopy(optimization)
        self.configure_metrics()
        self.configure_meta_keys()
        self.model.pop('evaluation')
        pipeline = self.model.pop('pipeline', None)
        if pipeline is not None:
            pipeline = Compose(pipeline)
        self.model = build_arch(self.model)
        if pipeline is not None:
            self.model = pipeline(self.model)
        if self.hparams.model.get('dataloader', None) is not None:
            self.hparams.model.pop('dataloader')

    def forward(self, x):
        # for testing
        return self.model.forward_test(x)['output']

    def training_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_train(input, label)
        loss = output['loss']

        # logging all output
        for name, value in output.items():
            if name != 'loss':
                self.log(f'train_{name}', value, on_step=True, on_epoch=True, prog_bar=False)
        # logging lr
        for opt in self.trainer.optimizers:
            # dirty hack to get the name of the optimizer
            self.log(f"lr-{type(opt).__name__}", opt.param_groups[0]['lr'], prog_bar=True, on_step=True, on_epoch=False)
        # logging loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_test(input, label)
        # return some useful meta-data for metrics calculation
        meta_data = [{k:input[k][idx] for k in self.meta_keys} for idx in range(label.shape[0])] \
            if self.meta_keys and isinstance(self.meta_keys, list) else None

        # logging all output
        for name, value in output.items():
            if name != 'output' and name != 'loss':
                self.log(f'val_{name}', value, on_step=True, on_epoch=True, prog_bar=False)
        # logging loss
        self.log("val_loss", output['loss'], on_step=True, on_epoch=True, prog_bar=True)
        return {'meta_data': meta_data, 'pred': output['output'], 'label': label}

    def validation_epoch_end(self, validation_step_outputs):
        # [val_step_output1, val_step_output2, ...]
        meta_data = [out for outputs in validation_step_outputs for out in outputs['meta_data']] \
            if validation_step_outputs[0]['meta_data'] else None
        pred = torch.cat([out['pred'] for out in validation_step_outputs], dim=0)
        label = torch.cat([out['label'] for out in validation_step_outputs], dim=0)

        # Report Metrics
        for metric in self.metrics:
            res = metric(pred, label) if getattr(metric, 'by', None) is None else metric(pred, label, meta_data)
            self.log(f"val_{metric.metric_name}", res, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_test(input, label)
        # some useful meta-data for metrics calculation
        meta_data = [{k: input[k][idx] for k in self.meta_keys} for idx in range(label.shape[0])] \
            if self.meta_keys and isinstance(self.meta_keys, list) else None

        # logging all output
        for name, value in output.items():
            if name != 'output' and name != 'loss':
                self.log(f'test_{name}', value, on_step=True, on_epoch=True, prog_bar=False)
        # logging loss
        self.log("test_loss", output['loss'], on_step=True, on_epoch=True, prog_bar=True)
        return {'meta_data': meta_data, 'pred': output['output'], 'label': label}

    def test_epoch_end(self, test_step_outputs):
        # [val_step_output1, val_step_output2, ...]
        meta_data = [out for outputs in test_step_outputs for out in outputs['meta_data']] \
            if test_step_outputs[0]['meta_data'] else None
        pred = torch.cat([out['pred'] for out in test_step_outputs], dim=0)
        label = torch.cat([out['label'] for out in test_step_outputs], dim=0)

        # Report Metrics
        for metric in self.metrics:
            res = metric(pred, label) if getattr(metric, 'by', None) is None else metric(pred, label, meta_data)
            self.log(f"test_{metric.metric_name}", res, on_step=False, on_epoch=True, prog_bar=True)


    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if getattr(self, 'warmup_scheduler', None) is not None:
            with self.warmup_scheduler.dampening():
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def configure_optimizers(self):
        # infer the maximun number of epochs and steps
        if self.optimization.get('type', 'epoch') == 'epoch':
            max_epochs = self.optimization.max_iters
            max_steps = self.num_max_steps
        else:
            assert self.optimization.get('type') == 'step', "optimization type must be epoch or step"
            max_steps = self.optimization.max_iters
            max_epochs = self.num_max_epochs
        # optimizer
        optim_cfg = deepcopy(self.optimization.optimizer)
        optim_cfg.model = self.model
        # New optimizer Ranger21
        # optim_cfg.update({'max_epochs': max_epochs, 'max_steps': max_steps}) if optim_cfg.type == 'Ranger21' else None
        optimizer = build_optimizer(optim_cfg)

        # scheduler
        if self.optimization.get('scheduler', None) is not None:
            scl_cfg = deepcopy(self.optimization.scheduler)
            scl_cfg.max_epochs = max_epochs
            scl_cfg.max_steps = max_steps
            scheduler = {"interval": scl_cfg.pop("interval", "step"),
                         'frequency': scl_cfg.pop("frequency", 1),
                         "monitor": scl_cfg.pop("monitor", "train_loss")}

            # warmup
            warmup_cfg = deepcopy(scl_cfg.pop('warmup', None))
            if warmup_cfg is not None:
                period = warmup_cfg.get('period', 0.1)
                if not isinstance(period, float):
                    period = period / max_epochs if self.optimization.get('type', 'epoch') == 'epoch' else period / max_steps
                warmup_cfg.period = int(period * max_steps) if scheduler.get('"interval"', 'step') == 'step' else int(period * max_epochs)
                warmup_cfg.optimizer = optimizer
                self.warmup_scheduler = build_scheduler(warmup_cfg)
            scl_cfg.optimizer = optimizer
            scheduler.update({"scheduler": build_scheduler(scl_cfg)})
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

    def configure_metrics(self):
        metrics = deepcopy(self.model.evaluation.get('metrics', dict(type='TorchMetircs', metric_name='Accuracy')))
        self.metrics = []
        if isinstance(metrics, list):
            for metric in metrics:
                self.metrics.append(build_metric(metric))
        elif isinstance(metrics, dict):
            self.metrics.append(build_metric(metrics))
        else:
            raise TypeError(f"Metrics must be a list or a dict, received {type(metrics)} type!")

    def configure_meta_keys(self):
        metrics = deepcopy(self.model.evaluation.get('metrics', dict(type='TorchMetircs', metric_name='Accuracy')).copy())
        self.meta_keys = []
        if isinstance(metrics, list):
            for metric in metrics:
                self.meta_keys.append(metric.get('by', None))
        elif isinstance(metrics, dict):
            self.meta_keys.append(metrics.get('by', None))
        else:
            raise TypeError(f"Metrics must be a list or a dict, received {type(metrics)} type!")
        self.meta_keys = [key for key in self.meta_keys if key is not None]

    @property
    def num_max_epochs(self):
        # get max training epochs inferred from datamodule and devices
        if self.trainer.max_epochs:
            return self.trainer.max_epochs

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return self.trainer.max_steps // (batches // effective_accum)

    @property
    def num_max_steps(self):
        # get max training steps inferred from datamodule and devices
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs






