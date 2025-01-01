import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the heart disease dataset
df = pd.read_csv('heart.csv')

# Define features and target variable
# Use the provided column names
X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak']]
y = df['HeartDisease']  # Assuming 'target' is your target variable column

# Create a OneHotEncoder object
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit the encoder to your categorical features and transform them
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina']  # List your categorical columns
encoded_features = encoder.fit_transform(X[categorical_features])

# Create a DataFrame from the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Drop the original categorical features and concatenate the encoded features
X = X.drop(categorical_features, axis=1)
X = pd.concat([X, encoded_df], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
def main():
    st.title("Heart Disease Prediction App")

    # Get user input
    age = st.slider("Age", 18, 90, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # Replace with actual chest pain types
    trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)
    chol = st.number_input("Cholesterol", 100, 500, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])  # Replace with actual ECG results
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 200, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 2.0)

    # Prepare input data for prediction
    user_input = pd.DataFrame({
        'Age': [age],
        'Sex': 1 if sex == "Male" else 0,  # Assuming 1 for Male, 0 for Female
        'ChestPainType': [cp],
        'RestingBP': [trestbps],
        'Cholesterol': [chol],
        'FastingBS': 1 if fbs == "Yes" else 0,  # Assuming 1 for Yes, 0 for No
        'RestingECG': [restecg],
        'MaxHR': [thalach],
        'ExerciseAngina': 1 if exang == "Yes" else 0,  # Assuming 1 for Yes, 0 for No
        'Oldpeak': [oldpeak]
    })# Ensure categorical features in user_input are strings
    for feature in categorical_features:
      user_input[feature] = user_input[feature].astype(str)

# Apply one-hot encoding to user input
    encoded_user_input = encoder.transform(user_input[categorical_features])
    encoded_user_input_df = pd.DataFrame(encoded_user_input, columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical features and concatenate encoded features
    user_input = user_input.drop(categorical_features, axis=1)
    user_input = pd.concat([user_input, encoded_user_input_df], axis=1)

    # Make prediction
    prediction = model.predict(user_input)[0]

    # Display prediction
    if prediction == 1:
        st.success("The model predicts that the person has a high risk of heart disease.")
    else:
        st.success("The model predicts that the person has a low risk of heart disease.")

    # Display accuracy
    st.write(f"Model Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    main()
