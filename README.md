
# MediCare AI
### AI-Powered Diabetes and Heart Disease Risk Prediction & Patient Monitoring System

MediCare AI is a machine learning-based healthcare project designed to demonstrate diabetes and heart disease risk classification using patient health parameters. It provides a simple web interface to enter health information, view model predictions, and review previous assessment records.

> **Disclaimer:** This project is an educational prototype, not a clinically validated medical system. Predictions must not be used for diagnosis, treatment, or medical decisions.

## Features

- Diabetes risk classification using machine learning
- Heart disease risk classification using machine learning
- Patient health parameter input forms
- Prediction results displayed through a web interface
- Assessment history stored using SQLite
- Model evaluation reports
- Separate frontend, backend, ML, data, and model folders

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Machine Learning | scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |
| Database | SQLite |
| Dataset Format | CSV |

## Project Structure

```text
MediCareAI/
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── diabetes.html
│   ├── heart.html
│   ├── history.html
│   ├── style.css
│   └── script.js
├── backend/
│   ├── app.py
│   ├── database.py
│   └── requirements.txt
├── ml/
│   ├── train_diabetes.py
│   ├── train_heart.py
│   └── predict.py
├── data/
│   ├── diabetes.csv
│   └── heart.csv
├── models/
├── reports/
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or a compatible Python version
- pip
- A web browser
- Diabetes and heart disease CSV datasets in the expected formats

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MediCareAI.git
cd MediCareAI
```

Replace `YOUR_USERNAME/MediCareAI` with your actual GitHub repository path.

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
pip install scikit-learn joblib
```

### 4. Add the datasets

Place the required CSV files in the `data/` folder:

- `data/diabetes.csv`
- `data/heart.csv`

The diabetes dataset must contain the expected eight input features and an `Outcome` column.

The heart dataset must contain the expected 13 input features and a `target` column, or 14 columns in the expected Cleveland dataset order.

### 5. Train the machine learning models

```bash
python ml/train_diabetes.py
python ml/train_heart.py
```

The generated model files are saved in `models/`, and evaluation reports are saved in `reports/`.

### 6. Start the application

```bash
python backend/app.py
```

Open this address in your browser:

http://127.0.0.1:5000

## Application Pages

- **Home:** Project introduction and navigation
- **Dashboard:** Access the available assessments
- **Diabetes Assessment:** Enter health parameters and view the model output
- **Heart Disease Assessment:** Enter heart-related health parameters and view the model output
- **Assessment History:** View saved assessment records

## Machine Learning Workflow

1. Load the CSV dataset.
2. Clean invalid and missing values.
3. Split the data into training and testing sets.
4. Impute missing values and scale the features.
5. Train a Logistic Regression classification model.
6. Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.
7. Save the trained model and use it for predictions through the Flask backend.

## Database

SQLite stores basic assessment type, predicted class, and timestamp. The current prototype does not implement patient authentication or comprehensive healthcare data protection. Do not enter real patient information.

## Limitations

- Model performance depends on the dataset and its quality.
- Public datasets may not represent every population.
- Model evaluation metrics do not establish clinical validity.
- This prototype is not intended for real-world clinical use.

## Future Enhancements

- Secure patient registration and authentication
- Improved patient monitoring dashboards
- Visualizations of assessment history
- Additional validated datasets and model comparisons
- Privacy, security, and access-control improvements
- Deployment after appropriate testing and review

## Author

Developed as an educational machine learning and web development project.

## License

This project is intended for educational and learning purposes.
