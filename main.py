# Imports necessary libraries
import csv
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QHBoxLayout, QScrollArea
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from collaborative import CollaborativeModel
import torch

# Loads the models
cfModel = CollaborativeModel(671, 9066)
cfModel.load_state_dict(torch.load('model/collaborative_model.pth'))

# Loads the data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Merges the datasets
data = pd.merge(ratings, movies, on="movieId")

# Creates mappings
movie_to_index = {movie_id: i for i, movie_id in enumerate(data['movieId'].unique())}
user_to_index = {user_id: i for i, user_id in enumerate(data['userId'].unique())}

def predict_with_model(model, input_data):
    input_data = torch.from_numpy(input_data)
    input_data = input_data.long()
    return model(input_data)

def recommend_movies(genre, user_id):
    # Ensures that the user ID exists
    if user_id not in user_to_index:
        raise ValueError(f"User ID {user_id} does not exist.")

    genre = genre.lower()
    genre_movies = movies[movies['genres'].str.lower().str.contains(genre)]
    valid_movie_ids = set(movie_to_index.keys())
    genre_movies = genre_movies[genre_movies['movieId'].isin(valid_movie_ids)]

    user_ratings = ratings[ratings['movieId'].isin(genre_movies['movieId'])]
    user_indices = user_ratings[user_ratings['userId'].isin(user_to_index.keys())]['userId'].map(user_to_index)
    movie_indices = genre_movies['movieId'].map(movie_to_index)

    features = []
    for user_id, movie_id in zip(user_indices, movie_indices):
        features.append([user_id, movie_id])
    features = np.array(features)

    cf_recommendations = predict_with_model(cfModel, features)
    cf_recommendations = np.squeeze(cf_recommendations.detach().numpy())

    top_movie_indices = cf_recommendations.argsort()[::-1][:15]
    top_movies = genre_movies.iloc[top_movie_indices]['title'].values

    return top_movies

class MovieRecommenderUI(QWidget):
    def __init__(self, parent=None):
        super(MovieRecommenderUI, self).__init__(parent)
        self.setWindowTitle("Movie Recommender")
        self.setGeometry(300, 300, 900, 700)

        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(p)

        self.title_font = QFont('Arial', 28)
        self.intro_font = QFont('Arial', 15)
        self.text_font = QFont('Arial', 15)
        self.title_font.setBold(True)
        self.intro_font.setItalic(True)
        self.text_font.setBold(True)

        self.layout = QVBoxLayout()
        self.layout.setSpacing(10) 
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        self.layout.setSpacing(10) 
        scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_area.setWidget(scroll_content)

        self.setLayout(self.layout)
        self.layout.addWidget(self.scroll_area)

        self.title_label = QLabel("Movie Recommender")
        self.title_label.setFont(self.title_font)
        self.title_label.setStyleSheet('color: purple;')
        self.title_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(self.title_label)

        self.intro_label = QLabel("Welcome to our movie recommendation application! We hope that we can help you find the perfect movie to watch next. Just enter your user ID and your favourite genre and we will provide you with 15 movies that we think you will love! Please feel free to provide feedback, so that we can provide you with better options next time.")
        self.intro_label.setFont(self.intro_font)
        self.intro_label.setStyleSheet('color: white;')
        self.intro_label.setAlignment(Qt.AlignCenter) 
        self.intro_label.setWordWrap(True)
        scroll_layout.addWidget(self.intro_label)

        user_layout = QHBoxLayout()
        self.user_label = QLabel("Enter your User ID:")
        self.user_label.setFont(self.text_font)
        self.user_label.setStyleSheet('color: white')
        self.user_entry = QLineEdit("1")
        user_layout.addWidget(self.user_label)
        user_layout.addWidget(self.user_entry)
        scroll_layout.addLayout(user_layout)
        
        genre_layout = QHBoxLayout()
        self.genre_label = QLabel("Select your favourite genre:")
        self.genre_label.setFont(self.text_font)
        self.genre_label.setStyleSheet('color: white')
        self.genre_entry = QComboBox()
        self.genre_entry.addItems(["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
        genre_layout.addWidget(self.genre_label)
        genre_layout.addWidget(self.genre_entry)
        scroll_layout.addLayout(genre_layout)

        self.recommend_button = QPushButton("Recommend")
        self.recommend_button.setStyleSheet('background-color: purple; color: white')
        scroll_layout.addWidget(self.recommend_button)

        self.recommendation_labels = []
        self.feedback_sliders = []
        self.submit_buttons = []

        self.recommend_button.clicked.connect(self.on_button_click)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet('color: red')
        scroll_layout.insertWidget(3, self.error_label)

    def create_feedback_handler(self, movie_title, feedback_slider):
        def feedback_handler():
            self.on_feedback_button_click(movie_title, feedback_slider)
        return feedback_handler

    def on_button_click(self):
        genre = self.genre_entry.currentText()
        user_id = self.user_entry.text()

        # Clears the previous error message if it exists
        self.error_label.setText("")

        # Ensures that the user ID is a number
        if not user_id.isdigit():
            self.error_label.setText("User ID must be a number.")
            return

        user_id = int(user_id)
        try:
            recommendations = recommend_movies(genre, user_id)

            # Removes previous recommendations, feedback sliders, and submit buttons
            for label in self.recommendation_labels:
                label.deleteLater()
            self.recommendation_labels.clear()

            for slider in self.feedback_sliders:
                slider.deleteLater()
            self.feedback_sliders.clear()

            for button in self.submit_buttons:
                button.deleteLater()
            self.submit_buttons.clear()

            # Generates new recommendations, feedback sliders and submit buttons
            for i, movie_title in enumerate(recommendations, start=1):
                movie_text = f"Movie {i}: {movie_title}"
                label = QLabel(movie_text)
                label.setStyleSheet('color: white')
                self.recommendation_labels.append(label)

                feedback_slider = QSlider(Qt.Horizontal)
                feedback_slider.setMinimum(1)
                feedback_slider.setMaximum(5)
                feedback_slider.setValue(5)
                feedback_slider.setTickPosition(QSlider.TicksBelow)
                feedback_slider.setTickInterval(1)
                self.feedback_sliders.append(feedback_slider)

                submit_button = QPushButton("Submit Feedback")
                submit_button.clicked.connect(self.create_feedback_handler(movie_title, feedback_slider))
                self.submit_buttons.append(submit_button)

                self.scroll_area.widget().layout().addWidget(label)
                self.scroll_area.widget().layout().addWidget(feedback_slider)
                self.scroll_area.widget().layout().addWidget(submit_button)

        except ValueError as e:
            self.error_label.setText(str(e))
            error_label = QLabel(str(e))
            self.recommendation_labels.append(error_label)
            self.scroll_area.widget().layout().addWidget(error_label)

    def on_feedback_button_click(self, movie_title, feedback_slider):
        user_id = int(self.user_entry.text()) if self.user_entry.text().isdigit() else 1
        movie_id = None
        if len(movies[movies['title'] == movie_title]) > 0:
            movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
        else:
            print(f"No movie found with title {movie_title}")

        if movie_id is not None:
            rating = feedback_slider.value()

            with open('data/feedback.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_id, movie_id, rating])

            print(f"Received feedback for {movie_title}: {rating} stars")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    movie_recommender_ui = MovieRecommenderUI()
    movie_recommender_ui.showMaximized()
    movie_recommender_ui.show()
    sys.exit(app.exec_())