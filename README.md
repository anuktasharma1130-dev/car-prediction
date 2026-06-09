## 📊 Project Output & Results

### 1. Model Evaluation Plot
Below is the evaluation plot comparing the **Actual Car Selling Prices** against the **Predicted Car Selling Prices**. 

The proximity of the blue data points to the red diagonal regression line indicates that the model has high accuracy and captures the price trends exceptionally well.

<p align="center">
  <img src="images/Screenshot 2026-06-09 191106.png" alt="Actual vs Predicted Car Selling Prices" width="80%">
</p>

### 2. Key Insights from the Plot
* **High Accuracy for Low to Mid-Range Cars:** The model aligns almost perfectly with the actual prices for vehicles priced between 0 to 10 Lakhs/Units.
* **Slight Variance in Premium Cars:** For higher-priced luxury cars (above 15 Units), the model shows a minor underestimation, which is common due to fewer premium car samples in the dataset.
* **Overall Performance:** The tight cluster of points around the ideal line proves the robustness of our regression algorithm.

### 3. 🔮 Future Improvements
* **Hyperparameter Tuning:** Fine-tune the model using GridSearchCV or RandomizedSearchCV to boost accuracy.
* **Web Deployment:** Deploy the model using Streamlit or Flask so users can input values through a UI.
* **More Features:** Include more features like car condition, number of previous owners, and insurance status.

### 4. 💡 How to Use the Model

You can load the trained model and make predictions like this:

```python
import pickle

# Load the saved model
model = pickle.load(open('car_price_model.pkl', 'rb'))

# Predict price for a new car [Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner]
predicted_price = model.predict([[2018, 7.5, 45000, 1, 0, 0, 0]])
print(f"Predicted Price: {predicted_price[0]}")
