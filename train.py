# Load Node Property Prediction datasets in OGB
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import TensorDataset, DataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader

from torch import nn
from torch.optim import Adam

from src.gnn import MultilayerGCN
from nntoolbox.utils import get_device
from nntoolbox.utils import save_model


NUM_EPOCHS = 1
WEIGHTS_PATH = "weights/model.pt"

dataset = DglNodePropPredDataset(name='ogbn-arxiv')
split_idx = dataset.get_idx_split()

g, labels = dataset[0]
# get split labels
train_label = dataset.labels[split_idx['train']]
valid_label = dataset.labels[split_idx['valid']]
test_label = dataset.labels[split_idx['test']]


train_idx = split_idx["train"]
valid_idx = split_idx["valid"]
test_idx = split_idx["test"]


train_dataset = TensorDataset(g.ndata["feat"][train_idx], labels[train_idx])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)


train_dataloader = NodeDataLoader(
    g, train_idx, MultiLayerFullNeighborSampler(2),
    batch_size=128,
    shuffle=True,
    drop_last=False,
    num_workers=1
)

valid_dataloader = NodeDataLoader(
    g, valid_idx, MultiLayerFullNeighborSampler(2),
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=1
)

test_dataloader = NodeDataLoader(
    g, test_idx, MultiLayerFullNeighborSampler(2),
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=1
)


model = MultilayerGCN(128, [64], 40).to(get_device())
optimizer = Adam(model.parameters())

criterion = nn.CrossEntropyLoss().to(get_device())

val_accs = []

for e in range(NUM_EPOCHS):
    model.train()

    for input_nodes, output_nodes, blocks in train_loader:
        optimizer.zero_grad()

        blocks = [b.to(get_device()) for b in blocks]
        input_features = blocks[0].srcdata['features']
        output_labels = blocks[-1].dstdata['label']

        output_predictions = model(blocks, input_features)
        loss = criterion(output_labels, output_predictions)

        loss.backward()
        optimizer.step()

    model.eval()

    total_cnt = 0
    acc_preds = 0


    for input_nodes, output_nodes, blocks in valid_dataloader:
        blocks = [b.to(get_device()) for b in blocks]
        input_features = blocks[0].srcdata['features']
        output_labels = blocks[-1].dstdata['label']

        output_predictions = model(blocks, input_features).argmax(-1)
        acc_preds += (output_predictions == output_labels).sum().item()
        total_cnt += input_features.shape[0]

    val_acc = acc_preds / total_cnt
    val_accs.append(val_acc)

    print("Validation Accuracy for Epoch {}: {}".format(e, val_acc))

    if val_acc == max(val_accs):
        save_model(model, WEIGHTS_PATH)
        print("Best validation accuracy so far. Saving model at {}".format(WEIGHTS_PATH))


total_cnt = 0
acc_preds = 0

for input_nodes, output_nodes, blocks in test_dataloader:
    blocks = [b.to(get_device()) for b in blocks]
    input_features = blocks[0].srcdata['features']
    output_labels = blocks[-1].dstdata['label']

    output_predictions = model(blocks, input_features).argmax(-1)
    acc_preds += (output_predictions == output_labels).sum().item()
    total_cnt += input_features.shape[0]

print("Test Accuracy: {}".format(acc_preds / total_cnt))
