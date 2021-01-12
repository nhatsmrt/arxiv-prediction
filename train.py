from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from dgl import add_self_loop

from torch import nn, no_grad
from torch.optim import Adam

from src.gnn import MultilayerGCN
from nntoolbox.utils import save_model, load_model, get_device

NUM_EPOCHS = 1
PRINT_LOSS_EVERY = 200
WEIGHTS_PATH = "weights/model.pt"

dataset = DglNodePropPredDataset(name='ogbn-arxiv')
split_idx = dataset.get_idx_split()

g, labels = dataset[0]
g = add_self_loop(g)
labels = labels.squeeze().to(get_device())

train_idx = split_idx["train"]
valid_idx = split_idx["valid"]
test_idx = split_idx["test"]

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

model = MultilayerGCN(128, [256], 40).to(get_device())
print(model)
optimizer = Adam(model.parameters())

criterion = nn.CrossEntropyLoss().to(get_device())

val_accs = []

for e in range(NUM_EPOCHS):
    model.train()

    for i, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        optimizer.zero_grad()

        blocks = [b.to(get_device()) for b in blocks]
        input_features = blocks[0].srcdata['feat']  # == g.ndata["feat"][input_nodes]
        output_labels = labels[output_nodes]  # == labels[blocks[-1].dstdata["_ID"]]

        output_predictions = model(blocks, input_features)
        loss = criterion(output_predictions, output_labels)

        if i % PRINT_LOSS_EVERY == 0:
            print("Loss at iteration {}: {}".format(i, loss.item()))

        loss.backward()
        optimizer.step()


    with no_grad():
      model.eval()

      total_cnt = 0
      acc_preds = 0

      for input_nodes, output_nodes, blocks in valid_dataloader:
          blocks = [b.to(get_device()) for b in blocks]
          input_features = blocks[0].srcdata['feat']
          output_labels = labels[output_nodes]

          output_predictions = model(blocks, input_features).argmax(-1)
          acc_preds += (output_predictions == output_labels).sum().item()
          total_cnt += output_nodes.shape[0]

      val_acc = acc_preds / total_cnt
      val_accs.append(val_acc)

      print("Validation Accuracy for Epoch {}: {}".format(e, val_acc))

      if val_acc == max(val_accs):
          save_model(model, WEIGHTS_PATH)
          print("Best validation accuracy so far. Saving model at {}".format(WEIGHTS_PATH))

with no_grad():
  total_cnt = 0
  acc_preds = 0

  load_model(model, WEIGHTS_PATH)
  model.eval()

  for input_nodes, output_nodes, blocks in test_dataloader:
      blocks = [b.to(get_device()) for b in blocks]
      input_features = blocks[0].srcdata['feat']
      output_labels = labels[output_nodes]

      output_predictions = model(blocks, input_features).argmax(-1)
      acc_preds += (output_predictions == output_labels).sum().item()
      total_cnt += output_nodes.shape[0]

  print("Test Accuracy: {}".format(acc_preds / total_cnt))