import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 64

img_transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5,),(0.5,))]) 

train_set = torchvision.datasets.MNIST(root = '../../data', train= True, transform= img_transform, download= True)
test_set = torchvision.datasets.MNIST(root = '../../data', train= False, transform= img_transform, download= True)

img, _ = train_set[0]
print(img.shape)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()
    self.f1 = nn.Linear(28*28,196)
    self.f2 = nn.Linear(196,49)
    self.f3 = nn.Linear(49,7)

  def forward(self, image):
    out = F.relu(self.f1(image))
    out = F.relu(self.f2(out))
    z = F.relu(self.f3(out))
    return z

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    self.f1 = nn.Linear(7,49)
    self.f2 = nn.Linear(49,196)
    self.f3 = nn.Linear(196,28*28)

  def forward(self, z):
    out = F.relu(self.f1(z))
    out = F.relu(self.f2(out))
    out = torch.tanh(self.f3(out))
    return out
 
class Autoncoder(nn.Module):
  def __init__(self):
    super(Autoncoder,self).__init__()
    self.encoder = Encoder()
    self.Decoder = Decoder()

  def forward(self, image):
    z = self.encoder(image)
    out =  self.Decoder(z)
    return out

  def train(self,model, train_loader, Epochs, loss_fn):
    train_loss_avg = []
    for epoch in range(Epochs):
      train_loss_avg.append(0)
      num_batches = 0
    
      for img, _ in train_loader:
          img = img + torch.randn(img.size()) * 0.01 + 0.1
          img = img.view(img.size(0),-1)
          img = img.to(device)

          img_recon = model(img)
          loss = loss_fn(img_recon, img)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          
      train_loss_avg[-1] /= num_batches
      print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, Epochs, train_loss_avg[-1]))
    return train_loss_avg
    

learning_rate = 0.001
autoencoder = Autoncoder()
autoencoder.to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
#autoencoder.train()
loss_values = autoencoder.train(autoencoder, train_loader,20, loss)
  
fig = plt.figure()
plt.plot(loss_values)
plt.xlabel('Epochs')
plt.ylabel('Reconstruction error')
plt.show()

def Show(out, title = ''):
  print(title)
  out = out.permute(1,0,2,3)
  grilla = torchvision.utils.make_grid(out,10,5)
  plt.imshow(transform.ToPILImage()(grilla), 'jet')
  plt.show()

def Show_Weight(out):
  grilla = torchvision.utils.make_grid(out)
  plt.imshow(transform.ToPILImage()(grilla), 'jet')
  plt.show()

with torch.no_grad():
  print("irrumpo")
  iterator = iter(test_loader)
  image1,label = iterator.next()
  
  print("imagen leida")
  fig, ax = plt.subplots(figsize=(10, 10))
  Show_Weight(image1[1:10])
  plt.show()\


image1,label = iterator.next()
image1,label = iterator.next()
  

image = image1 + torch.randn(image1.size())*0.1 + 0.2 
  
 
print("imagen con ruido")
image = image.view(image.size(0),1,28,28)
fig, ax = plt.subplots(figsize=(10, 10))
Show_Weight(image[1:10])
plt.show()


image = image.to(device)
image = image.view(image.size(0),-1)

salida = autoencoder(image)
salida = salida.view(salida.size(0),1,28,28)

print("imagen reconstruida")
fig, ax = plt.subplots(figsize=(10, 10))
Show_Weight(salida[1:10])
plt.show()




z = autoencoder.encoder(image)
deviation,mean = torch.std_mean(z,axis=0)
new_z = (torch.randn(5,7).to(device)*deviation + mean).to(device)

print(new_z.shape)
new_image = autoencoder.Decoder(new_z)
new_image = new_image.view(5,1,28,28)
fig, ax = plt.subplots(figsize=(5, 5))
Show_Weight(new_image[1:5])
plt.show()
