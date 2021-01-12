from pytorch_lightning import LightningDataModule
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from dgl import add_self_loop
from dgl.dataloading.pytorch import _NodeDataLoaderIter


__all__ = ['ArxivDataModule']


class MyNodeDataLoaderIter(_NodeDataLoaderIter):
    def __iter__(self):
        return self


class MyNodeDataLoader(NodeDataLoader):
    def __iter__(self):
        """Return the iterator of the data loader."""
        if self.is_distributed:
            # Directly use the iterator of DistDataLoader, which doesn't copy features anyway.
            return iter(self.dataloader)
        else:
            return MyNodeDataLoaderIter(self)


class ArxivDataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        self.split_idx = dataset.get_idx_split()
        self.g, labels = dataset[0]
        self.g.ndata["label"] = labels.squeeze()
        self.g = add_self_loop(self.g)
        self.batch_size = batch_size

    def get_split(self, split, batch_size: int, shuffle: bool, num_workers: int):
        return MyNodeDataLoader(
            self.g, split, MultiLayerFullNeighborSampler(2),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers
        )

    def train_dataloader(self):
        return self.get_split(self.split_idx["train"], self.batch_size, True, 1)

    def val_dataloader(self):
        return self.get_split(self.split_idx["valid"], self.batch_size, False, 1)

    def test_dataloader(self):
        return self.get_split(self.split_idx["test"], self.batch_size, False, 1)

    def transfer_batch_to_device(self, batch, device):
        input_nodes, output_nodes, blocks = batch
        blocks = [b.to(device) for b in blocks]

        return (input_nodes, output_nodes, blocks)
