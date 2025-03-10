# Salary Prediction by Tenure Project

This project implements a machine learning model to predict salary ranges based on an employee's tenure (time at the company) and their professional level within the organization. The project analyzes whether employees with longer tenures tend to earn higher salaries.

## Project Overview

The application uses both linear regression and polynomial regression models to predict salaries based on:
- Time at the company (in months)
- Professional level within the company (1-10 scale)

The project consists of three main components:
1. A Jupyter notebook for data analysis and model training
2. A FastAPI service for model deployment
3. A Streamlit web interface for easy interaction with the model

## Dataset

The dataset used for training is located in `datasets/dataset_salario.csv`, containing information about employee tenure, professional level, and their corresponding salaries.

## Project Structure

```
ia-para-produtos-regressao-polinomial/
│
├── datasets/
│   └── dataset_salario.csv
│
├── api_modelo_salario.py        # FastAPI implementation
├── app_streamlit_salario.py     # Streamlit web application
├── modelo_salarios.ipynb        # Jupyter notebook for model development
│
└── modelo_salario.pkl           # Serialized model file
```

## Model Selection

The project implements and compares two regression models:
- Linear Regression: Used as a baseline model
- Polynomial Regression: Selected as the final model due to better performance in capturing the non-linear relationship between tenure/level and salary

## API Implementation

The FastAPI implementation (`api_modelo_salario.py`) provides an endpoint for salary prediction:

```python
@app.route('/predict')
def predict(data: request_body):
    input_features = {
        'tempo_na_empresa': data.tempo_na_empresa,
        'nivel_na_empresa': data.nivel_na_empresa
    }
    pred_df = pd.DataFrame(input_features, index=[1])
    y_pred = modelo_poly.predict(pred_df)[0].astype(float)
    return {'salario_em reais': y_pred.tolist()}
```

## Web Interface

The Streamlit application (`app_streamlit_salario.py`) provides an intuitive interface where users can:
- Input employee tenure using a slider (1-120 months)
- Select the professional level using a slider (1-10)
- Get an estimated salary prediction with a single button click

## Installation and Setup

1. Clone the repository
2. Install dependencies:

```bash
pipenv install scikit-learn scipy pandas matplotlib seaborn ipykernel pingouin fastapi pydantic streamlit uvicorn requests numpy
```

3. Start the API server:

```bash
uvicorn api_modelo_salario:app --reload
```

4. Launch the Streamlit interface:

```bash
streamlit run app_streamlit_salario.py
```

## Usage Example

1. Visit the Streamlit web interface (typically at http://localhost:8501)
2. Adjust the sliders for "Time at Company" and "Professional Level"
3. Click "Estimate Salary"
4. View the predicted salary

For API usage, send a POST request to `http://localhost:8000/predict` with the following JSON structure:

```json
{
  "tempo_na_empresa": 60,
  "nivel_na_empresa": 5
}
```

## Model Performance

The polynomial regression model was chosen as it better captures the non-linear relationship between tenure, professional level, and salary. Detailed analysis and model performance metrics can be found in the Jupyter notebook.

## Future Improvements

- Incorporate additional features like department, education level, and industry experience
- Implement model retraining pipeline for continuous improvement
- Add confidence intervals for predictions
- Extend the interface with visualizations of salary trends by tenure and level
