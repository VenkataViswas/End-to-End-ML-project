From Raw Data to Deployment: An End-to-End ML Project 🚀

This repository contains a clean, modular, end-to-end machine learning pipeline that takes raw data, processes it, trains and evaluates models, and prepares them for deployment.

The project is structured to follow best practices, enabling reproducibility, clarity, and CI/CD readiness for ML workflows.

----------------------------------------
📌 Project Highlights

- End-to-end pipeline: data ingestion → cleaning → feature engineering → training → evaluation
- Modular code structure for maintainability
- Logging for traceability
- Configuration management for flexible experimentation
- CI/CD using GitHub Actions with custom AWS EC2 runners
- Jupyter notebooks for EDA and experimentation
- Ready for MongoDB integration in future iterations

----------------------------------------
```plaintext 🗂️ Project Structure End-to-End-ML-Project/ | ├── .github/workflows/ # GitHub Actions workflows for CI/CD ├── config/ # Configuration files (YAML/JSON) ├── logs/ # Log files for pipeline runs ├── notebooks/ # Jupyter notebooks for EDA and experiments ├── src/ # Source code for ML pipeline │ ├── __init__.py │ ├── data_ingestion.py │ ├── data_transformation.py │ ├── model_trainer.py │ ├── model_evaluation.py │ └── utils.py ├── requirements.txt # Python dependencies ├── setup.py # Makes project pip-installable ├── app.py #  Entry point for starting the pipeline and serving the model ├── README.md # Project documentation └── .gitignore ```

----------------------------------------
⚙️ Features

- Data Ingestion: Load raw data, store intermediate files.
- Data Cleaning & Feature Engineering: Handle missing values, encode categorical variables, scale numerical features.
- Model Training & Evaluation: Train ML models and evaluate using cross-validation and performance metrics.
- Logging: Track pipeline steps, errors, and experiment metadata.
- CI/CD: Automatic linting, testing, and checks using GitHub Actions.
- Modularity: Easy to extend or replace pipeline components.
- Future Ready: MongoDB can be integrated for data storage and experiment tracking.

----------------------------------------
🚀 Getting Started

1️⃣ Clone the repository
    git clone https://github.com/VenkataViswas/End-to-End-ML-project.git
    cd End-to-End-ML-project

2️⃣ Create a virtual environment and activate it
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3️⃣ Install dependencies
    pip install -r requirements.txt

4️⃣ Run the flask application to start pipeline and open an webpage
    python application.py

----------------------------------------
📊 Exploratory Data Analysis

EDA notebooks are available in the notebooks/ folder to:
- Understand data distributions.


----------------------------------------
🚦 CI/CD Pipeline

- Configured using GitHub Actions with:
    - Standard runners for lightweight checks.
    - Custom AWS EC2 runners for heavier jobs.
- Automates code linting, testing, and ensures stable builds.

----------------------------------------

⭐ If you find this repository helpful, please consider starring it!
