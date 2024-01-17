# Imports necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle

# Parameters
num_epochs = 10  
learning_rate = 0.001
batch_size = 256  

class ContentModel(nn.Module):
    def __init__(self, input_dim):
        super(ContentModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layer(x).float()

    def recommend(self, user_features, top_n=15):
        # Prepares the tensor
        user_features_tensor = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0) # The unsqueeze operation makes sure that the tensor has the right shape

        self.eval()
        with torch.no_grad():
            user_features_tensor = user_features_tensor.to(next(self.parameters()).device)
            print("user_features_tensor shape: ", user_features_tensor.shape)
            scores = self(user_features_tensor)
            top_n = min(top_n, scores.size(0))
            _, indices = torch.topk(scores, top_n)
        return indices.tolist()
    
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    tags = pd.read_csv("data/tags.csv")

    # Merges the data
    data = pd.merge(ratings, movies, on="movieId")
    data = pd.merge(data, tags, on="movieId")

    # Removes duplicate columns
    data = data.loc[:,~data.columns.duplicated()]

    mlb = MultiLabelBinarizer()
    data['genres'] = data['genres'].str.split('|')
    data['tag'] = data['tag'].apply(lambda x: [x])

    all_genres = set()
    for genres in data['genres']:
        all_genres.update(genres)
        
    print(all_genres)
    print(data['genres'].head())

    # Applies multilabel binarizer
    genres = pd.DataFrame(mlb.fit_transform(data.pop('genres')), columns=mlb.classes_, index=data.index)

    genres = genres.loc[:,~genres.columns.duplicated()]

    tag_encoded = pd.DataFrame(mlb.fit_transform(data.pop('tag')), columns=mlb.classes_, index=data.index)
    tag_encoded = tag_encoded.loc[:,~tag_encoded.columns.duplicated()]

    data = pd.concat([data, genres, tag_encoded], axis=1)
    data = data.loc[:,~data.columns.duplicated()]

    with open('model/mlb_instance.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    return data

def augment_data(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise

def train_model(data, num_epochs, batch_size, learning_rate):
    X = data.drop(columns=["userId_x", "movieId", "rating", "timestamp_x", "title", "userId_y", "timestamp_y"]).values
    y = data["rating"].values 
    model = ContentModel(X.shape[1])
    criterion = nn.MSELoss()  
    optimizer = Adam(model.parameters(), lr=learning_rate)  
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    train_losses = []
    val_losses = []
    MAEs = []
    RMSEs = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_ids], X[val_ids]
        y_train, y_val = y[train_ids], y[val_ids]
        X_train = augment_data(X_train)
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
        val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))

        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        for epoch in range(num_epochs):
            model.train()
            train_losses_epoch = []
            for batch in train_loader:
                x_train_batch, y_train_batch = batch
                optimizer.zero_grad()
                outputs = model(x_train_batch)
                loss = criterion(outputs, y_train_batch.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_losses_epoch.append(loss.item())
            train_losses.append(np.mean(train_losses_epoch))
            scheduler.step()

            model.eval()
            with torch.no_grad():
                losses = []
                preds = []
                for batch in val_loader:
                    x_val_batch, y_val_batch = batch
                    outputs = model(x_val_batch)
                    loss = criterion(outputs, y_val_batch.view(-1, 1))
                    losses.append(loss.item())
                    preds.append(outputs.detach().numpy())
                val_losses.append(np.mean(losses))
                preds = np.concatenate(preds)
                MAEs.append(mean_absolute_error(y_val, preds))
                RMSEs.append(np.sqrt(mean_squared_error(y_val, preds)))
                print(f"Fold: {fold+1}, Epoch: {epoch+1}, Validation Loss: {np.mean(losses)}, MAE: {MAEs[-1]}, RMSE: {RMSEs[-1]}")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(MAEs, label='MAE')
    plt.plot(RMSEs, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()

    torch.save(model, 'model/contentbased_model.pth')

    return model

def main():
    # Loads and preprocesses the data
    data = load_data()
    input_dim = data.shape[1]
    print(f'Model input dimension: {input_dim}')
    
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    genre_start_index = data.columns.get_loc(genre_cols[0])
    genre_end_index = data.columns.get_loc(genre_cols[-1]) + 1
    
    print(f'Genre start index: {genre_start_index}')
    print(f'Genre end index: {genre_end_index}')

if __name__ == "__main__":
    main()
    data = load_data()
    model = train_model(data, num_epochs, batch_size, learning_rate)