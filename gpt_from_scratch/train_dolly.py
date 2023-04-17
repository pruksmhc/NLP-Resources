
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch
import os
from nano_model import GPT, GPTConfig
from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter()



def train_one_epoch(model, optimizer, train_dataloader, epoch_index, tb_writer, out_dir, config):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data['input_ids'], data['target']
        labels = labels[:, 1:]
        labels = torch.cat((labels, torch.full( (32, 1), 50257)), dim=1)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        _, loss = model(inputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        print(model.trasformer.wte.weight)
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': i,
                    'best_val_loss': last_loss,
                    'config': config,
                }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    return last_loss

def train(num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_from_disk('databricks_dolly')
    train_dataloader = DataLoader(ds['train'].with_format("torch"), batch_size=32)
    #test_dataloader = DataLoader(ds['test'].with_format("torch"), batch_size=32)

    config = GPTConfig()
    model = GPT(config, 50257)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_dataloader, epoch, tb_writer, "dolly1run", config)
        # TODO Add evaluation run.

train(2)