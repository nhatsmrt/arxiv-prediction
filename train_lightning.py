from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers


from src.data import ArxivDataModule
from src.models import *
from torch import nn

NUM_EPOCHS = 100
BATCH_SIZE = 128
WEIGHTS_PATH = "weights/model.pt"



if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(max_epochs=NUM_EPOCHS, logger=tb_logger)
    datamodule = ArxivDataModule(BATCH_SIZE)
    model = GraphLightningModule(MultilayerGCN(128, [256], 40), nn.CrossEntropyLoss())

    trainer.fit(model=model, datamodule=datamodule)