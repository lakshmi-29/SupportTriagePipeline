__Support Ticket Triage Pipeline__  <br>(Scikit-Learn/Pandas Stack)
This project implements an automated machine learning pipeline to analyze and triage incoming support tickets. By utilizing scikit-learn for modeling and Pandas/NumPy for data handling, the pipeline aims to classify the Category of a ticket (e.g., Billing, Technical) and predict its Priority Score instantly upon arrival.

__Overview__
The core objective is to reduce manual triage time by applying two distinct ML models:Classification Model: 
- Predicts the ticket Category (e.g., Billing, Technical, Account) using Logistic Regression on TF-IDF features.Regression Model: Predicts a continuous Priority Score (e.g., $1.0$ to $5.0$) using Random Forest Regressor.
- This setup ensures tickets are immediately routed to the correct department and flagged by urgency. 

Prerequisites:
1. Ensure you have a suitable environment with:Python >3.8, Git
2. Installation - Clone the repository 
3. Install dependencies 


__Data Structure__
The pipeline assumes your raw ticket data is provided in a CSV format (support_tickets.csv) with the following required columns: 
| Column Name | Type | Role |
|----------|----------|----------|
| ticket_description    | Text (String)   | Input Feature   |
| category    | Categorical   | Auxiliary   |
| Department    | Categorical (String)   | Target1   |
| Resolution_Days    | Categorical (String)   | Target2   |


__Pipeline Tasks and Models__
The triage system utilizes a multi-task learning approach, trained and evaluated using scikit-learn.

3.1. Task 1: Department Classification (Routing)
This task focuses on determining which team should handle the ticket.

Goal: Predict the discrete value of the Department column.

Model: Logistic Regression (sklearn.linear_model).

Evaluation: classification_report (Precision, Recall, F1-Score).


Task 2: Time-to-Resolution Regression (Prioritization)
This task aims to predict the estimated effort/time needed for resolution, which serves as a crucial prioritization signal.

Goal: Predict the continuous value of the Resolution_Days column.

Model: Random Forest Regressor (sklearn.ensemble).

Evaluation: mean_squared_error (MSE).


__Core Implementation Steps__
The pipeline relies on these steps, typically executed within a single Python script (e.g., triage_pipeline.py).

1. Data Loading and Cleaning
Uses Pandas (pd.read_csv) to load data and NumPy for any necessary numerical transformations.

Drops missing values and prepares text features.

2. Feature Engineering
Text Vectorization: The Ticket_Description text is converted to numerical features using TfidfVectorizer from sklearn.feature_extraction.text.

3. Data Splitting
The data is split into training and testing sets using train_test_split (sklearn.model_selection) to ensure unbiased model evaluation.

4. Training and Evaluation
Both the Logistic Regression and Random Forest Regressor models are trained and evaluated, and the results are printed to the console.


