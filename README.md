# Accident_Traffic_Duration

![image](https://user-images.githubusercontent.com/72606788/212783541-0a8a9878-410c-4e21-add2-a12620ab2ac9.png)

## Overview

This repo's main feature is a model trained on the [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) that predicts the length of time an accident will impact traffic given features that can be known immediately (weather conditions, day/time, nearby road features). Individual predictions can be made and evaluated via the Flask app (app.py), and further discussion is included on the web app.

To read more about this project, check out my Medium article [here!](https://medium.com/@brendonkirouac/predicting-traffic-accident-duration-with-python-4488aaca5c4a)

## Usage

### Libraries

- numpy
- pandas
- xgboost
- joblib
- json
- plotly
- flask
- matplotlib
- seaborn
- sklearn

For any usage, it is recommended to install the required packages - requirements files are included for pip and conda package managers.
If using pip, create a new environment through your environment manager if needed to avoid updating system package versions!

  1. `cd` into top-level directory for this repo
  2. `pip install -r pipreqs.txt` for pip, `conda create --name <env> --file requirements.txt` for conda
     - **Note**: if on a non-windows system, comment out pywin32 & pywinty lines, otherwise this will error

For the web app:
1. Clone this repo
2. `cd` into the top-level directory
3. `flask run`
4. Visit the address shown in the CLI ([127.0.0.1:5000](http://127.0.0.1:5000/))
5. Fill out the fields in the webpage to generate a prediction given different input features

For data_processing.ipynb:
1. Download the [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) and extract to the top-level directory of this repo
2. `jupyter lab` via CLI, or preferred interface for python notebooks
3. Work through the cells top-down

For model_exploration.ipynb:
1. Generate the `cleaned_data.csv` from data_processing.ipynb
1. Work through the cells! The GridSearchCV cell will take a long time, so I would recommend skipping unless modifying to develop a stronger model.

### Files

- 📁**templates**
   - **index.html** # main page of web app. Contains write-up about the data analysis & model.
- 📁**static**
   - contains static images generated by the notebooks for use on the webpage & medium article.
- **app.py** # Python script to run webpage using Flask
- **data_processing.ipynb** # Jupyter notebook used to process initial csv into cleaned_data.csv, and generate plots found in static folder
- **model_exploration.ipynb** # Jupyter notebook used to train models from cleaned_data.csv
- **grid_search_results.csv** # CSV file that contains results of the grid_search training, with relevant parameters
- **states_ordered.csv** # Contains U.S. states ordered by mean accident duration, calculated from the dataset. Used in web app for form submission.
- **classifier.pkl** # XGBoost softprob classifier object generated by model_exploration.ipynb and used in run.py
- **requirements.txt** # requirements file to use for conda environments
- **pipreqs.txt** # requirements file to use for pip environments
- **README.md** # This file!

### Results

Using xgboost's softprobability classifier with gridsearchCV for hyperparameter tuning, we are able to achieve an accuracy score of 0.63 (0.64 on the test set.)
The most performant parameters were learning_rate: 0.1, max_depth: 18, n_estimators: 200.

### Acknowledgements
- U.S. Accidents Dataset
  - [Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019. ](https://arxiv.org/abs/1906.05409)
  - [Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.](https://arxiv.org/abs/1909.09638)
