# Imports necessary libraries
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from math import sqrt

from contentBased import load_data as load_content_data, ContentModel
from collaborative import load_data as load_collaborative_data, CollaborativeModel

# Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 256
weight_decay = 1e-4

class HybridModel(nn.Module):
    def __init__(self, contentModel, collabModel, content_weight=0.5):
        super(HybridModel, self).__init__()
        self.contentModel = contentModel
        self.collabModel = collabModel
        self.content_weight = content_weight

    def forward(self, X_content, X_collab):
        out_content = self.contentModel(X_content)
        out_collab = self.collabModel(X_collab)
        # Weighted sum of the two models' outputs
        return self.content_weight * out_content + (1 - self.content_weight) * out_collab

if __name__ == "__main__":
    # Loads the data
    data_content = load_content_data()
    data_collaborative = load_collaborative_data()

    required_columns = ["userId_x", "movieId", "rating", "timestamp_x", "title", "userId_y", "timestamp_y"]
    if all(col in data_content.columns for col in required_columns):
        X_content = data_content.drop(columns=required_columns).values
        y_content = data_content["rating"].values
    else:
        raise KeyError("Required columns not found in data_content")

    X_collaborative_train, X_collaborative_val, X_collaborative_test, y_collaborative_train, y_collaborative_val, y_collaborative_test, num_users, num_movies, user_encoded2user, movie_encoded2movie, movieId_to_title = data_collaborative

    # Constructs the hybrid datasets
    train_data = TensorDataset(torch.tensor(X_content[:len(X_collaborative_train)], dtype=torch.float), torch.tensor(X_collaborative_train, dtype=torch.long), torch.tensor(y_content[:len(y_collaborative_train)], dtype=torch.float))
    val_data = TensorDataset(torch.tensor(X_content[len(X_collaborative_train):-len(X_collaborative_test)], dtype=torch.float), torch.tensor(X_collaborative_val, dtype=torch.long), torch.tensor(y_content[len(y_collaborative_train):-len(y_collaborative_test)], dtype=torch.float))
    test_data = TensorDataset(torch.tensor(X_content[-len(X_collaborative_test):], dtype=torch.float), torch.tensor(X_collaborative_test, dtype=torch.long), torch.tensor(y_content[-len(y_collaborative_test):], dtype=torch.float))

    # Creates dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)  # test dataloader

    # Loads the models
    contentModel = torch.load('model/contentbased_model.pth')
    collabModel = CollaborativeModel(num_users, num_movies)

    # Initialises the model, loss function, optimizer
    model = HybridModel(contentModel, collabModel)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses = []
    val_losses = []
    MAEs = []  # Mean Absolute Errors
    RMSEs = []  # Root Mean Squared Errors

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for i, (X_content_batch, X_collab_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_content_batch, X_collab_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_losses.append(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            val_losses_batch = []
            MAEs_batch = []
            RMSEs_batch = []
            for i, (X_content_batch, X_collab_batch, y_batch) in enumerate(val_loader):
                outputs = model(X_content_batch, X_collab_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_losses_batch.append(loss.item())
                MAEs_batch.append(mean_absolute_error(y_batch.numpy(), outputs.squeeze().numpy()))
                RMSEs_batch.append(sqrt(mean_squared_error(y_batch.numpy(), outputs.squeeze().numpy())))

            val_losses.append(np.mean(val_losses_batch))
            MAEs.append(np.mean(MAEs_batch))
            RMSEs.append(np.mean(RMSEs_batch))
            
        scheduler.step()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(MAEs, label='MAE')
    plt.plot(RMSEs, label='RMSE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')

    plt.show()

    torch.save(model.state_dict(), 'model/hybrid_model.pth')