# Imports necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.nn import Dropout
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import sqrt

# Parameters
num_epoch = 200
batch_size = 256
embedding_dim = 150
weight_decay = 0.01
learning_rate = 0.005

class CollaborativeModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=150):
        super(CollaborativeModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(embedding_dim * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.7)
        self.dense2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        self.dense3 = nn.Linear(64, 32) 
        self.dense4 = nn.Linear(32, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        vector = torch.cat([user_vector, movie_vector], dim=-1)
        vector = self.flatten(vector)
        vector = self.dropout1(torch.relu(self.bn1(self.dense1(vector)))) 
        vector = self.dropout2(torch.relu(self.bn2(self.dense2(vector)))) 
        vector = torch.relu(self.dense3(vector)) 
        return self.dense4(vector)
    
    def recommend(self, user_id, all_movie_ids, top_n=15):
        self.eval()
        with torch.no_grad():
            user_id_tensor = torch.tensor([user_id] * len(all_movie_ids)).reshape(-1, 1)
            movie_id_tensor = torch.tensor(all_movie_ids).reshape(-1, 1)
            user_movie_tensor = torch.cat((user_id_tensor, movie_id_tensor), 1)
            scores = self(user_movie_tensor)
            _, indices = torch.topk(scores.squeeze(), top_n)
        return indices.tolist()

def load_data():
    # Loads and merges the data
    movies = pd.read_csv("data/movies.csv", encoding="Latin1")
    ratings = pd.read_csv("data/ratings.csv")

    feedbacks = pd.read_csv("data/feedback.csv", header=0, names=["userId", "movieId", "rating"])
    feedbacks['userId'] = feedbacks['userId'].astype(int)
    feedbacks['movieId'] = feedbacks['movieId'].astype(int)
    feedbacks['rating'] = feedbacks['rating'].astype(float)

    ratings = pd.concat([ratings, feedbacks])

    data = pd.merge(movies, ratings, on="movieId", how='inner')

    # Preprocesses the data
    user_ids = data["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(sorted(user_ids))}
    user_encoded2user = {i: x for i, x in enumerate(user_ids)}

    movie_ids = data["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(sorted(movie_ids))}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    data["user"] = data["userId"].map(user2user_encoded)
    data["movie"] = data["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    data = data.sample(frac=1, random_state=42)
    x = data[["user", "movie"]].values
    y = data["rating"].values

    # Creates a mapping from movieId to title
    movieId_to_title = pd.Series(movies.title.values, index=movies.movieId).to_dict()

    # Splits the data into train, validation, and test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.3, random_state=42)  # Increase validation set size

    return x_train, x_val, x_test, y_train, y_val, y_test, num_users, num_movies, user_encoded2user, movie_encoded2movie, movieId_to_title

def train_model(num_epoch=num_epoch, batch_size=batch_size, learning_rate=learning_rate):
    # Loads the data
    x_train, x_val, x_test, y_train, y_val, y_test, num_users, num_movies, _, _, _ = load_data()

    # Prepares PyTorch Datasets and Dataloaders
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    val_data = TensorDataset(torch.tensor(x_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float))
    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float))

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    train_losses = []
    val_losses = []
    MAEs = []  # Mean Absolute Errors
    RMSEs = []  # Root Mean Squared Error

    # Initialises the model
    model = CollaborativeModel(num_users, num_movies, embedding_dim)

    # Initialises the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Trains the model
    for epoch in range(num_epoch):
        model.train()
        losses_epoch = []
        for batch in train_loader:
            x_train_batch, y_train_batch = batch
            optimizer.zero_grad()
            outputs = model(x_train_batch)
            loss = criterion(outputs, y_train_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            losses_epoch.append(loss.item())
        train_losses.append(np.mean(losses_epoch))

        model.eval()
        with torch.no_grad():
            losses = []
            preds = []
            for batch in val_loader:
                x_val_batch, y_val_batch = batch
                outputs = model(x_val_batch)
                loss = criterion(outputs, y_val_batch.unsqueeze(1))
                losses.append(loss.item())
                preds.append(outputs.detach().numpy())
            avg_loss = np.mean(losses) 
            val_losses.append(avg_loss)  
            preds = np.concatenate(preds)
            MAEs.append(mean_absolute_error(y_val, preds))
            RMSEs.append(sqrt(mean_squared_error(y_val, preds)))
            print(f"Epoch: {epoch+1}, Loss: {avg_loss}, MAE: {MAEs[-1]}, RMSE: {RMSEs[-1]}")

        scheduler.step()  

    # Saves the model
    torch.save(model.state_dict(), 'model/collaborative_model.pth')

    # Plots losses
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plots metrics
    plt.subplot(2, 1, 2)
    plt.plot(MAEs, label='MAE')
    plt.plot(RMSEs, label='RMSE Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()