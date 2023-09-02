# Machine Learning Project Template

## Template Structure


```
project/
├── data/
│   ├── raw/                            # Raw data 
│   ├── processed/                      # Data after preprocessing
│   ├── engineered/                     # Data after feature engineering 
│   └── final/                          # Data split into train, validation, and test sets 
│
├── notebooks/                          # Jupyter notebooks directory for experimentation
│
├── src/
│   ├── preprocessing/                  # Data processing module
│   ├── feature_engineering/            # Feature engineering module
│   ├── training/                       # Training module
│   ├── evaluation/                     # Evaluation module
│   ├── visualization/                  # Visualization module (utility functions for plotting)
│   ├── utils/                          # Utility module (useful functions)
│   └── main.py                         # Main script for running the whole pipeline
│
├── tests/                              # Unit tests directory
│
├── config/                             # Configuration files directory
|
|── reports/                            # Reports on different experiments
|
├── conda-dev.yaml                      # Conda dependencies
│
└── README.md              # Project documentation
```


## Setting up a Virtual Environment

The conda-dev.yaml file contains useful libraries for data science and machine learning. Use it to create a conda environment with all the dependencies needed to run the project.

```bash
conda env create -f conda-dev.yaml
```

Activate the environment

```bash
conda dev-env
```

To add a new package to the environment, add it to the `conda-dev.yaml` file, then update the environment:

```bash
conda env update --file conda-dev.yaml
```

## Running the Project

The order of the pipeline is as follows:

1. Preprocessing: Clean the data
2. Feature Engineering: Add new features
3. Data Splitting: Split the data into train, validation, and test sets
3. Training: Train the model and tune it on the validation set
4. Evaluation: Evaluate the model on the test set
