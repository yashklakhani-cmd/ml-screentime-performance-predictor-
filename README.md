# ml-screentime-performance-predictor-
A Scikit-Learn Machine Learning model utilizing multiple linear regression to predict academic outcomes based on screen time and study habits.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_and_predict():
    print("--- Academic Performance ML Predictor ---")
    
    # 1. Mock Dataset (Features: Study Hours, Screen Time | Target: Exam Score)
    data = {
        'Study_Hours': [2, 3, 4, 5, 1, 6, 3, 4],
        'Screen_Time_Hours': [5, 4, 3, 2, 6, 1, 4, 2],
        'Exam_Score': [65, 72, 80, 88, 55, 95, 70, 85]
    }
    df = pd.DataFrame(data)
    
    # 2. Split Data into Features (X) and Target (y)
    X = df[['Study_Hours', 'Screen_Time_Hours']]
    y = df['Exam_Score']
    
    # 3. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse:.2f}\n")
    
    # 6. Make a New Prediction
    # Scenario: 4 hours of study, strictly lessening screen time to 1.5 hours
    new_student = pd.DataFrame({'Study_Hours': [4], 'Screen_Time_Hours': [1.5]})
    predicted_score = model.predict(new_student)
    
    print("--- New Prediction ---")
    print(f"Input -> Study: 4 hrs, Screen Time: 1.5 hrs")
    print(f"Predicted Exam Score: {predicted_score[0]:.2f}")

if __name__ == "__main__":
    train_and_predict()
