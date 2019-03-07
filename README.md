# human_robot_interaction_eeg

Study of human robot interaction with robot under admittance control and monitoring of subjects using eeg

## Project Organization

├── LICENSE
├── Makefile
├── README.md
├── docs
│  
├── models
│   └── trained_model.pth
├── notebooks
│   ├── CNN.ipynb
│   ├── EEG_Data.ipynb
│   ├── Play_Ground.ipynb
│   ├── Unit_Tests.ipynb
│   └── Utility_Functions.ipynb
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── config.yml
│   ├── data
│   │   ├── __init__.py
│   │   ├── clean_eeg_dataset.py
│   │   ├── create_eeg_dataset.py
│   │   ├── create_robot_dataset.py
│   │   ├── create_torch_dataset.py
│   │   ├── eeg_utils.py
│   │   └── robot_utils.py
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── networks.py
│   │   ├── predict_model.py
│   │   ├── train_model.py
│   │   └── utils.py
│   └── visualization
│       ├── __init__.py
│       └── robot_position_plot.py
├── test_environment.py
├── tests
│   ├── __init__.py
│   └── test.py
└── tox.ini
* * *

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
