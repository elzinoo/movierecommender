# Deep Learning-Based Movie Recommendation System 

This is a movie recommendation system implemented using Python and the PyQt5 library. It uses collaborative filtering techniques to generate recommendations for users. Users are given 15 personalised movie recommendations based on their favourite genre and user ID.


## Installation

To install this project, you will need to have Python 3.9 installed on your machine. A virtual environment was used for this project, so you can activate it, and it will have all the necessary packages installed.

To activate the virtual environment on a macOS or Linux, navigate to the directory of the virtual environment and run the following command in your terminal: 
source myenv/bin/activate

If you are using a Windows machine, you will need to recreate the virtual environment and install the following packages:

- pandas                       2.0.1
- PyQt5                        5.15.9
- numpy                        1.24.3
- matplotlib                   3.7.1
- mpmath                       1.3.0
- scikit-learn                 1.2.2
- torch                        2.0.1
- watchdog                     3.0.0


## Usage

The models from the collaborative filtering and content-based filtering have been saved, so there is no need to run them before running the main.py file. The main.py file runs the user interface and can be run normally. 

The application will open and display the movie recommendation user interface.

To use the interface, enter a user ID, select your favourite genre from the dropdown menu, and click the "Recommend" button.
The system will generate a list of 15 movies based on the user ID and genre selected and display them on the screen.
To provide feedback, the slider can be used (from 1 star to 5 stars), and then you have to click on the "submit Feedback" button for each movie you want to provide feedback on. 


## Additional Information

The movie data is loaded from the movies.csv file, and the user ratings are loaded from the ratings.csv file.
The collaborative filtering model used for generating recommendations is trained and stored in the collaborative_model.pth file in the 'model' folder.
The system handles errors such as invalid user IDs and missing movie titles.
Feedback provided by users is stored in the feedback.csv file for future improvements to the recommendation system.
The monitor.py file detects changes to the feedback.csv file and reruns the CollaborativeModel.

The combine.py file is an attempt at creating a hybrid model for this system.

For future development, this system should use a weighted combination of collaborative and content based filtering, save the model and use that to make recommendations. 