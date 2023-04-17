import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def test_gpt_implementation(model, config):
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
  running_loss = 0.0
  inputs = torch.range(1, 15).long()
  inputs = torch.reshape(inputs,(1,15 )) 
  labels = inputs + 1
  for i in range(10):
      # get the inputs; data is a list of [inputs, labels]
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs, loss = model.forward(inputs,labels)
      loss.backward()
      optimizer.step()
      print(model.pos_embed)
      print("step ",i ," loss: ", loss.item())
      writer.add_scalar("loss/train", loss)
  inputs = torch.range(1, 10).to(int)
  inputs = torch.reshape(inputs,(1,10 )) 
  model.inference(inputs, 12)