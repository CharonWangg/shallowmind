import torch
import pytorch_lightning as pl
from shallowmind.src.model.builder import build_arch, build_optimizer, build_scheduler, build_metric


class ModelInterface(pl.LightningModule):
    def __init__(self, model, optimization):
        super().__init__()
        self.save_hyperparameters()
        self.configure_metrics()
        self.configure_meta_keys()
        model.pop('evaluation')
        self.model = build_arch(model)

    def training_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_train(input, label)
        loss = output['loss']
        # logging loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_test(input, label)
        # return some useful meta-data for metrics calculation
        meta_data = [{k:input[k][idx] for k in self.meta_keys} for idx in range(label.shape[0])] \
            if self.meta_keys and isinstance(self.meta_keys, list) else None

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

    def configure_optimizers(self):
        # optimizer
        optim_cfg = self.hparams.optimization.optimizer.copy()
        optim_cfg['model'] = self.model
        optimizer = build_optimizer(optim_cfg)

        # scheduler
        scl_cfg = self.hparams.optimization.scheduler.copy()
        scheduler = {"interval": scl_cfg.pop("interval", "step"),
                     "monitor": scl_cfg.pop("monitor", "val_loss")}
        scl_cfg['optimizer'] = optimizer
        scl_cfg['max_epochs'] = self.hparams.optimization.max_epochs
        scl_cfg['max_iters'] = self.hparams.optimization.max_iters
        scheduler.update({"scheduler": build_scheduler(scl_cfg)})

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_metrics(self):
        metrics = self.hparams.model.evaluation.get('metrics', dict(type='TorchMetircs', metric_name='Accuracy')).copy()
        self.metrics = []
        if isinstance(metrics, list):
            for metric in metrics:
                self.metrics.append(build_metric(metric))
        elif isinstance(metrics, dict):
            self.metrics.append(build_metric(metrics))
        else:
            raise TypeError(f"Metrics must be a list or a dict, received {type(metrics)} type!")

    def configure_meta_keys(self):
        metrics = self.hparams.model.evaluation.get('metrics', dict(type='TorchMetircs', metric_name='Accuracy')).copy()
        self.meta_keys = []
        if isinstance(metrics, list):
            for metric in metrics:
                self.meta_keys.append(metric.get('by', None))
        elif isinstance(metrics, dict):
            self.meta_keys.append(metrics.get('by', None))
        else:
            raise TypeError(f"Metrics must be a list or a dict, received {type(metrics)} type!")
        self.meta_keys = [key for key in self.meta_keys if key is not None]






