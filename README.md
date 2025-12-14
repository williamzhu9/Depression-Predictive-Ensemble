# Depression-AnxietyPredictiveModel
This project features an ensemble of machine learning models with the intent of early detection of depression or risk of depression in people. Specifically, this project trains random forest and XGBoost models on existing patient datasets to classify new patients as depressed or not depressed. We use a feature-partitioned ensemble, which splits some input I of patients with all features, into various subsets of features designated for each model. For example: model a is trained on feature subset A, model b is trained on feature subset B, while the input I is the union of subsets A and B. The outputted predictions from each of the models are then evaluated using a weighted voting system to determine if a patient in the overall input I should be classified as depressed or not. This project functions as a proof-of-concept of a much larger idea of automated early detection/alerting of mental illnesses. While the project might feature some design and realism limitations, conceptually it provides a good foundation for further research to expand upon. 

# Technologies Used
To develop this project, we based it in python using Pandas as the backbone of our data processing parts. As for our models, we used the Sci-Kit library to use their random forest and XGboost models along with additional tools (such as the confusion matrices and correlation charts).

# Directory Tree
```
project-root/
│
├── main.py
├── README.md
│
├── raw/
│   └── input/
│       └── input.csv
│
├── output/
│   └── ensemble_final_predictions.csv
│
├── models/
│   └── models_saved/
│       ├── model_depression_anxiety_rf.pkl
│       ├── model_depression_anxiety_xg.pkl
│       ├── model_student_depression_rf.pkl
│       └── model_student_depression_xg.pkl
│
├── scripts/
│   ├── __init__.py
│   ├── depression_anxiety_processor.py
│   └── student_depression_processor.py
│
├── requirements.txt
└── .gitignore
```

---

## Descriptions

### `main.py`
Entry point for running the ensemble and entire pipeline. Loads trained models, partitions and preprocesses input data, performs weighted voting, and outputs predictions.

### `raw/input/input.csv`
Raw input data provided, defined manually using real-life represenations and generalizations

### `output/`
Contains generated prediction results.
- `ensemble_final_predictions.csv`: Final predictions with confidence scores.

### `models/`
Directory containing all machine learning models and model-related files
- models_saved: Contains all the exported models to be loaded and used by ensemble
- *_model.py: Machine learning model specified for a specific dataset and type

### `scripts/`
Directory containing all the necessary preprocessing scripts and virtual environment
- sklearn-env: Sci-Kit virtual environment
- *_processor.py: Preprocessing script specific to a certain dataset
- .gitattributes: Used to define file types for git large file storage

### `requirements.txt`
Python dependencies required to run the project.

### `.gitignore`
Files to exclude from version control

# Execution details
Currently, to simply run the program on the pre-configured input, follow these steps
1. Activate and set up the virtual environment: 
- From root, `CD` to scripts/
- Run the `sklearn-env/Scripts/activate` command
- Install dependencies using the `pip install -r requirements.txt` command
2. Execute main from root
- Return to root directory using the `cd ..` command (from the scripts directory)
- Run the `python main.py` command

Some things to note with the execution:
- The input is currently fixed (explicit csv file for input) and must contain a complete set of all the features.
- Output predictions are outputted both in the terminal as well as in a final csv for each record in the input. 
- The models are pretrained and the ensemble is preconfigured to the current models. To add/change the models it requires you to retrain the models and add them to the ensemble if necessary.

# Other details
For more specific details regarding this project and the implementation, please read our report.