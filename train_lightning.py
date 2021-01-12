from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from src.data import ArxivDataModule
from src.models import *
from torch import nn

NUM_EPOCHS = 100
BATCH_SIZE = 128
WEIGHTS_PATH = "weights/model.pt"


if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(filepath=WEIGHTS_PATH, monitor='val_acc', mode='max', save_top_k=1)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(max_epochs=NUM_EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback])
    datamodule = ArxivDataModule(BATCH_SIZE)
    model = GraphLightningModule(MultilayerGCN(128, [256], 40), nn.CrossEntropyLoss())

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(GraphLightningModule.load_from_checkpoint(
        checkpoint_path="{}.ckpt".format(WEIGHTS_PATH),
        model=MultilayerGCN(128, [256], 40),
        criterion=nn.CrossEntropyLoss()
    ), datamodule=datamodule)
