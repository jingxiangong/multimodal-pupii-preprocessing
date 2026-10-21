# multimodal-pupil-preprocessing
This project provides a guideline for processing pupil diameter data and integrating it with emotion expression data from multimodal datasets from iMotions10 (version 10.1.38911.4). This project uses python programming lauguage.

Project Structure
'Pupil_Preprocessing_Functions.py' contains functions for cleaning, filtering, interpolating, normalizing, and integrating pupil diameter measurements. 
'Example.py' shows how to use those functions.

Development Environment
Python version: 3.11.11
Key libraries: pandas (2.2.3), NumPy (2.2.4), SciPy (1.15.2), scikit-learn (1.6.1), matplotlib (3.10.1)
Raw data exported from iMotion 10 (version 10.1.38911.4)

Installation
Install Python if not already installed.
Install required libraries using pip and the provided requirements.txt file:
pip install -r requirements.txt
If you do not have a requirements.txt file, you can install the libraries individually:
pip install pandas==2.2.3 numpy==2.2.4 scipy==1.15.2 scikit-learn==1.6.1 matplotlib==3.10.1

Usage
Place your data file(e.g. data.csv, exported from iMotions 10) in the project directory
Run Example.py to process your data 

Input and Output
Input: CSV file(e.g. data.csv) exported from iMotion 10, containing pupil diameter or other multimodal measurements.
Output: Result.csv -cleaned and integrated datasets, ready for further analysis or visualization.

All code is explained line by line within thescripts to assist users in understanding and customizing the workflow.
