import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, jsonify, request

# Tahap 1: Pengumpulan Data
data = pd.read_csv('sales_and_rentals_data.csv')
print("Data yang dikumpulkan:")
print(data.head())

# Tahap 2: Data Cleaning
print("\nCleaning Data...")
# Cek missing values
print("Missing values per column:")
print(data.isnull().sum())
# Untuk contoh ini, data sudah bersih

# Tahap 3: Data Transformation
print("\nTransforming Data...")
# Menambahkan kolom revenue (pendapatan)
data['revenue'] = data['quantity'] * data['price']
print(data.head())

# Tahap 4: Exploratory Data Analysis (EDA)
print("\nExploratory Data Analysis...")
# Statistik Deskriptif
print(data.describe())

# Visualisasi dengan barplot
plt.figure(figsize=(12, 8))
sns.barplot(x='product_name', y='revenue', hue='gender', data=data)
plt.title('Revenue per Product by Gender')
plt.show()

# Visualisasi dengan pie chart
print("\nCreating Pie Chart...")
# Total revenue per product
revenue_per_product = data.groupby('product_name')['revenue'].sum()

# Pie chart
plt.figure(figsize=(8, 8))
revenue_per_product.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Revenue Distribution by Product')
plt.ylabel('')
plt.show()

# Visualisasi berdasarkan jenis transaksi
plt.figure(figsize=(12, 8))
sns.barplot(x='product_name', y='revenue', hue='transaction_type', data=data)
plt.title('Revenue per Product by Transaction Type')
plt.show()

# Tahap 5: Modelling Data
print("\nModelling Data...")
# Prediksi revenue berdasarkan quantity dan transaction_type
# Convert categorical data to numerical
data['transaction_type'] = data['transaction_type'].apply(lambda x: 1 if x == 'Sale' else 0)
data = pd.get_dummies(data, columns=['gender'], drop_first=True)

X = data[['quantity', 'transaction_type', 'gender_Male']]
y = data['revenue']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Tahap 6: Validasi dan Tuning Model
print("\nValidasi dan Tuning Model...")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot hasil prediksi
plt.scatter(X_test['quantity'], y_test, color='blue', label='Actual')
plt.scatter(X_test['quantity'], y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Tahap 7: Interpretasi dan Penyajian Hasil
print("\nInterpretasi dan Penyajian Hasil...")
print(f"Model Linear Regression:\nIntercept: {model.intercept_}\nCoefficients: {model.coef_}")

# Tahap 8: Deployment dan Monitoring
# Untuk deploy, menggunakan Flask sebagai contoh
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    quantity = data['quantity']
    transaction_type = 1 if data['transaction_type'] == 'Sale' else 0
    gender_Male = 1 if data['gender'] == 'Male' else 0
    prediction = model.predict(np.array([[quantity, transaction_type, gender_Male]]))
    return jsonify({'revenue': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

# Tahap 9: Maintenance dan Iterasi
# Maintenance dan iterasi melibatkan pemantauan kinerja model dan memperbaruinya dengan data baru secara berkala.
