import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Input
X = data[['Customers']]

# Multiple Outputs
y = data[['Bread', 'Cake', 'Cookies']]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict(X)

# Example prediction
result = model.predict([[85]])
print("For 85 customers:")
print("Bread:", result[0][0])
print("Cake:", result[0][1])
print("Cookies:", result[0][2])

# Graphs for each item
plt.figure(figsize=(10,6))

plt.scatter(X, data['Bread'], label='Bread Actual')
plt.plot(X, predictions[:,0], label='Bread Predicted')

plt.scatter(X, data['Cake'], label='Cake Actual')
plt.plot(X, predictions[:,1], label='Cake Predicted')

plt.scatter(X, data['Cookies'], label='Cookies Actual')
plt.plot(X, predictions[:,2], label='Cookies Predicted')

plt.xlabel("Customers")
plt.ylabel("Items")
plt.title("Bakery Multi-Item Prediction")
plt.legend()

plt.savefig("output.png")
plt.show()
# Residuals for Bread
residuals_bread = data['Bread'] - predictions[:,0]

plt.figure()
plt.scatter(predictions[:,0], residuals_bread)
plt.axhline(y=0)
plt.title("Bread Residual Plot")
plt.savefig("bread_residual.png")
plt.show()
# Residuals for Cake
residuals_cake = data['Cake'] - predictions[:,0]

plt.figure()
plt.scatter(predictions[:,0], residuals_cake)
plt.axhline(y=0)
plt.title("Cake Residual Plot")
plt.savefig("cake_residual.png")
plt.show()
# Residuals for Cookies
residuals_cookies = data['Cookies'] - predictions[:,0]

plt.figure()
plt.scatter(predictions[:,0], residuals_cookies)
plt.axhline(y=0)
plt.title("Cookies Residual Plot")
plt.savefig("cookies_residual.png")
plt.show()