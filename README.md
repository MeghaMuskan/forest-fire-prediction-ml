# Forest Fire Prediction using Machine Learning

A machine learning project that predicts forest fire risk using regression models, built into a tested, deployed Flask web application.

**🔗 Live app:** [forest-fire-megha.azurewebsites.net](http://forest-fire-megha.azurewebsites.net)

---

## 📊 Dataset
- Algerian Forest Fire Dataset
- Features include:
  - Temperature
  - Humidity
  - Wind Speed
  - Rainfall
  - FFMC, DMC, ISI (fire-weather sub-indices)
  - Classes, Region

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Flask, gunicorn
- pytest
- Azure App Service (deployment)

---

## 🧠 Models Implemented
- Linear Regression
- Ridge Regression
- Lasso Regression
- RidgeCV
- LassoCV
- ElasticNet

**Ridge Regression** was selected for production after cross-validation showed it generalized best on this dataset, balancing bias and variance without the sparsity tradeoffs introduced by Lasso/ElasticNet.

---

## 🔍 Project Workflow
1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Selection
4. Model Training
5. Hyperparameter Tuning (RidgeCV, LassoCV)
6. Model Evaluation and Comparison
7. REST API Development (Flask)
8. Unit Testing (pytest)
9. Deployment (Azure App Service)

---

## 📈 Results
- Compared performance of six regression approaches
- Observed impact of regularization (Ridge & Lasso)
- Identified key environmental factors affecting forest fire risk
- Selected Ridge Regression for production based on cross-validated performance

---

## 🧪 Testing

The Flask app is covered by a 21-case pytest suite, including:
- Route availability and method handling
- Exact risk-category boundary behavior (e.g. a score of 4.99 vs. 5.0)
- Gauge percentage scaling and overflow capping
- Missing-field and invalid-input handling

Run locally:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## 📁 Project Structure
```
forest-fire-prediction-ml/
│
├── application.py                          # Flask app and prediction route
├── requirements.txt
│
├── models/
│   ├── ridge.pkl                            # Trained Ridge regression model
│   └── scaler.pkl                           # StandardScaler used for preprocessing
│
├── templates/
│   └── home.html                            # Main UI
│
├── static/
│   └── css/style.css                        # Styling
│
├── tests/
│   ├── conftest.py                          # Test fixtures and mocked model
│   └── test_application.py                  # 21-case pytest suite
│
├── csv_files/                               # Raw and cleaned datasets
│
├── 01-model-training.ipynb
├── 02-linear-regression.ipynb
├── 03-ridge-lasso-regression.ipynb
│
└── README.md
```

---

## 🌐 Running Locally
```bash
git clone https://github.com/MeghaMuskan/forest-fire-prediction-ml.git
cd forest-fire-prediction-ml
pip install -r requirements.txt
python application.py
```
Visit `http://localhost:5000`.

---

## ☁️ Deployment

Deployed on **Azure App Service** (Linux, Python 3.11 runtime) using `gunicorn` as the production WSGI server:
```bash
az webapp up --runtime PYTHON:3.11 --name forest-fire-megha --resource-group forest-fire-rg --sku F1
az webapp config set --resource-group forest-fire-rg --name forest-fire-megha \
  --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 application:app"
```

---

## 🎯 Objective
To analyze environmental data and build predictive models that can help in understanding and estimating forest fire risk — and to deliver that model as a usable, tested, production web application.

---

## 🚀 Future Improvements
- ~~Deployment as a web application~~ ✅ Done — live on Azure
- ~~Unit testing~~ ✅ Done — 21-case pytest suite
- ~~Hyperparameter tuning~~ ✅ Done — RidgeCV/LassoCV used to select optimal alpha
- CI/CD pipeline for automated testing on push
- Model optimization with additional environmental features
- Use of advanced ML models (e.g. gradient boosting)

---

## 👩‍💻 Author
**Megha Muskan**
B.Tech Computer Science Student | KIIT University
[GitHub](https://github.com/MeghaMuskan)
