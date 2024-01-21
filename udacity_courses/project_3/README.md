# PROJECT 3: Disaster Response Project


## Project Motivation
The aim of this project is to build a model for an API that classifies disaster messages. The dataset contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that the messages can be sent to an appropriate disaster relief agency. The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## File Descriptions
notebooks/: contains the notebooks used for the project.

web_application/: contains the files needed for the web app, including the data, the model, and the code for the web app.

## Installations
In the repository's [README.md](../../README.md) file, you can find the python version used for this project (3.10.6), and how to install the dependencies using poetry.


## Instructions
To process data:
1. Run the following command in `udacity_courses/project_3/web_application`to set up your database and model data.
    - To run ETL pipeline that cleans data and stores in database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. Run the following command in `udacity_courses/project_3/web_application/` to train the model and save it.
    - To run ML pipeline that trains classifier and saves: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`, which will generate `udacity_courses/project_3/web_application/models/classifier.pkl`

To run the web app:
1. Run the following command in the app's directory to run your web app.
    - `python run.py`
