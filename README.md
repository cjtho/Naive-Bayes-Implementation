# Naive Bayes Classifier Project

## Project Goal
The aim of this project is to build a Naive Bayes classifier from scratch, fine-tune the model to find the best estimator, and analyse the performance of the estimators. The process involved understanding the intricacies of the algorithm and applying it to a real-world dataset.

## Dataset Overview
The dataset, consisting of research paper abstracts, was sourced from Kaggle ([Research Paper Abstracts](https://www.kaggle.com/datasets/nikhilmittal/research-paper-abstracts/)). For practicality, the dataset was condensed to 10,000 entries from the original 85,409 by combining the class labels and abstracts into a single file for easier manipulation and to improve computational efficiency.

## Code Structure
- `NaiveBayesPerformanceAnalysis.ipynb`: The Jupyter notebook that documents the creation, tuning, and performance analysis of the Naive Bayes classifier.
- `naive_bayes.py`: The Python script with the implementation of the Naive Bayes classifier.
- `model_calculator.py`: A supplementary script to determine the number of models for the random search.
- `text_flattener.py`: A utility script for text data formatting and standardisation.
- `science_field_dataset.csv`: The curated dataset file used in this project.

## Usage
Begin with `NaiveBayesPerformanceAnalysis.ipynb` to understand the workflow. Ensure the `science_field_dataset.csv` is in the working directory and requisite Python packages are installed. The supplementary scripts `text_flattener.py` and `model_calculator.py` are incorporated within the notebook to facilitate certain functions.

## Contact
Email: christopherjthomaswork@gmail.com
