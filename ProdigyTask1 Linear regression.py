#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


# Load the dataset (replace with actual file path)
data = pd.read_csv(r"C:\Users\ACER\Downloads\train.csv")


# In[3]:


# Select relevant features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]


# In[4]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=58)


# In[5]:


# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[6]:


# Make predictions
y_pred = model.predict(X_test)


# In[7]:


# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared= False)
print(f"Root Mean Squared Error: {rmse:.2f}")


# In[8]:


#prediction for a new house
new_house = pd.DataFrame([[2000, 3, 2]], columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")


# In[9]:


#prediction for a new house
new_house_01 = pd.DataFrame([[2300, 3, 3]], columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
predicted_price_01 = model.predict(new_house_01)
print(f"Predicted price for the new house: ${predicted_price_01[0]:,.2f}")


# In[10]:


#prediction for a new house
new_house_02 = pd.DataFrame([[3000, 5, 4]], columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
predicted_price_02 = model.predict(new_house_02)
print(f"Predicted price for the new house: ${predicted_price_02[0]:,.2f}")


# In[13]:


#prediction for a new house
new_house_02 = pd.DataFrame([[4000, 9, 8]], columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
predicted_price_02 = model.predict(new_house_02)
print(f"Predicted price for the new house: ${predicted_price_02[0]:,.2f}")


# In[ ]:




