import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from matplotlib import pyplot

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor())
    
    figure = pyplot.figure(figsize=(5, 5))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = random.randint(0, len(train_data))
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        pyplot.title(label)
        pyplot.axis('off')
        pyplot.imshow(img.squeeze())
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05)
    pyplot.show()
    
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f'Feature batch shape: {train_features.shape}')
    print(f'Labels batch shape: {train_labels.shape}')
    # 删除维度为 1 的轴
    img = train_features[0].squeeze()
    label = train_labels[0]
    pyplot.imshow(img)
    pyplot.show()
    print('Label:', str(label))
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error.
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return total_loss / num_batches
    
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss
        
    epochs = 30
    train_his = []
    val_his = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_dataloader, model, loss_fn, optimizer)
        train_his.append(loss)
        loss = test(test_dataloader, model, loss_fn)
        val_his.append(loss)

    pyplot.plot(range(1, epochs + 1), train_his, label='Train Loss', color='blue', marker='o')
    pyplot.plot(range(1, epochs + 1), val_his, label='Validation Loss', color='red', marker='o')   
    pyplot.legend()
    pyplot.grid()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
    
    print("Done!")
