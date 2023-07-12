# Disaster Response Pipeline Project

## Introduction

This tool designed to assist individuals and organizations during disaster events.
The primary goal is to provide a practical solution that have a real-world impact, making a difference in times of crisis.

## Files in the Repository

- `app/`: Directory containing the web app files.
  - `template/`: Directory containing the HTML templates for the web app.
    - `master.html`: Main page of the web app.
    - `go.html`: Classification result page of the web app.
  - `run.py`: Flask file that runs the app.
  
- `data/`: Directory containing the data files and processing scripts.
  - `disaster_categories.csv`: Data to process.
  - `disaster_messages.csv`: Data to process.
  - `process_data.py`: Script for processing the data.
  - `DisasterResponse.db`: Database to save clean data to.

- `models/`: Directory containing the model training script and saved model.
  - `train_classifier.py`: Script for training the classifier.
  - `classifier.pkl`: Saved model.

- `README.md`: Documentation for the project.
- `requirements.txt`: List of Python packages required for this project.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
