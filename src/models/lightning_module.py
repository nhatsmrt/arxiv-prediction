from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import functional as FM

from torch.optim import Adam
from torch import nn


__all__ = ['GraphLightningModule']



class GraphLightningModule(LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module):
        super().__init__()
        self.model, self.criterion = model, criterion

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        input_features = blocks[0].srcdata['feat']  # == g.ndata["feat"][input_nodes]
        output_labels = blocks[-1].dstdata['label']

        output_predictions = self(blocks, input_features)
        loss = self.criterion(output_predictions, output_labels)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        input_features = blocks[0].srcdata['feat']
        output_labels = blocks[-1].dstdata['label']

        output_predictions = self(blocks, input_features)

        val_loss = self.criterion(output_predictions, output_labels)
        val_accs = FM.accuracy(pred=output_predictions, target=output_labels)

        metrics = {'val_acc': val_accs, 'val_loss': val_loss}
        self.log_dict(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)
