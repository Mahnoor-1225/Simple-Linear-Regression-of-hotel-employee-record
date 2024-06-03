import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_excel("C:/Users/mahno/OneDrive/Desktop/internship/your_dataset.xlsx", engine='openpyxl')

# Scatter plot between total bill and tip
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="total_bill", y="tip")
plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.show()

# You can create similar scatter plots for other variables as well

# Define independent and dependent variables
X = data[["total_bill"]]  # Independent variable
y = data["tip"]           # Dependent variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
