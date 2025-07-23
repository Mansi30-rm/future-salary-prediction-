import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("survey_results_public.csv", low_memory=False)

# Rename columns for simplicity
df = df.rename(columns={'YearsCodingProf': 'Experience', 'Salary': 'Salary'})

# Function to convert experience ranges to numeric values
def convert_experience(value):
    if pd.isna(value):
        return np.nan
    if "or more" in value:
        return 30
    if "-" in value:
        parts = value.replace(" years", "").split("-")
        return (int(parts[0]) + int(parts[1])) / 2
    if value == "0-2 years":
        return 1
    return np.nan

# Apply conversion to Experience column
df['Experience'] = df['Experience'].apply(convert_experience)

# Convert Salary to numeric (may be object type)
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# Drop rows where either Experience or Salary is NaN
df = df.dropna(subset=['Experience', 'Salary'])

# Filter out unrealistic salary values
df = df[(df['Salary'] > 10000) & (df['Salary'] < 300000)]

# Show cleaned data size
print(f"✅ Cleaned dataset size: {df.shape}")

# Prepare input (x) and output (y)
x = df['Experience'].values
y = df['Salary'].values

# Perform linear regression using NumPy
m, b = np.polyfit(x, y, 1)

# Print the equation
print(f"\n📈 Equation: Salary = {m:.2f} * Experience + {b:.2f}")

# Predict salary for a given experience
years = 5
predicted_salary = m * years + b
print(f"\n🔮 Predicted salary for {years} years of experience: ₹{predicted_salary:.2f}")

# 👤 Take user input for prediction
try:
    user_exp = float(input("\n👤 Enter your years of professional coding experience: "))
    predicted_user_salary = m * user_exp + b
    print(f"🔮 Predicted salary for {user_exp} years of experience: ₹{predicted_user_salary:.2f}")
except ValueError:
    print("⚠ Please enter a valid numeric value for experience.")
    predicted_user_salary = None

# 💰 Ask for current salary and compare
if predicted_user_salary is not None:
    try:
        actual_salary = float(input("💰 Enter your current salary (in ₹): "))
        diff = predicted_user_salary - actual_salary
        if diff > 0:
            print(f"📊 Your predicted salary is ₹{abs(diff):.2f} more than your current salary.")
        else:
            print(f"📊 Your current salary is ₹{abs(diff):.2f} more than the predicted salary.")
    except ValueError:
        print("⚠ Please enter a valid numeric value for salary.")

# Plotting the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5, label="Data Points")
plt.plot(x, m*x + b, color='red', label=f"Regression Line: Salary = {m:.2f} * Experience + {b:.2f}")
plt.xlabel('Years of Experience')
plt.ylabel('Salary (in ₹)')
plt.title('Salary vs Experience with Linear Regression')
plt.legend()
plt.show()