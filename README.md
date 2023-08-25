# Machine Learning Project Template

## Template Structure

```
project/
    data/
        raw/                            # Raw data 
        processed/                      # Data after preprocessing
        engineered/                     # Data after feature engineering 
        final/                          # Data split into train, validation, and test sets 
    notebooks/                          # Jupyter notebooks directory for experimentation
    src/
        preprocessing/                  # Data processing module
        feature_engineering/            # Feature engineering module
        training/                       # Training module
        evaluation/                     # Evaluation module
        visualization/                  # Visualization module (utility functions for plotting)
        utils/                          # Utility module (useful functions)
        main.py                         # Main script for running the whole pipeline
    tests/                              # Unit tests directory
    config/                             # Configuration files directory
    reports/                            # Reports on different experiments
    conda-dev.yaml                      # Conda dependencies
    README.md                           # Project documentation
```

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

Use the conda-dev.yaml file to create a conda environment with all the dependencies needed to run the project.

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

- 1. Preprocessing
- 2. Feature Engineering
- 3. Training
- 4. Evaluation
