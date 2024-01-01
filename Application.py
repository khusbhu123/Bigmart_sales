# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
def load_data():
  data = pd.read_csv('Bigmart_Train.csv')
  return data

# Label Encoding
def label_encode(data, columns):
  le = LabelEncoder()

  for col in columns:
    if col in data.columns:
     if data[col].dtype == 'object':
# For categorical columns, use LabelEncoder for ordinal encoding
       le.fit(data[col])
       data[col] = le.transform(data[col])

    return data

# Train the Random Forest Regressor Model
def train_model(X_train, y_train):
# Define a column transformer to handle numerical variables
  preprocessor = ColumnTransformer(
  transformers=[('num', SimpleImputer(), ['Item_Weight', 'Item_Visibility', 'Item_MRP']),
               ])

# Define the pipeline with Random Forest Regressor
  model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', RandomForestRegressor(
                             ))
                             ])

# Train the model
  model.fit(X_train, y_train)
  return model

# Preprocess the data
def preprocess_data(data):
  if data is None or data.empty:
      print("empty or None Input Data.")
      return pd.DataFrame()
  
# Filling the Missing Value for Item Weight Column    
  item_weight_mean = data.pivot_table(values="Item_Weight", index="Item_Identifier")
  
# We need to check which row has the missing values and store it
  miss_bool = data["Item_Weight"].isnull()
  
# Now let's fill the missing values
  for i, item in enumerate(data.get("Item_Identifier", [])):
     if miss_bool[i] and item in item_weight_mean:
      if item in item_weight_mean:
       data["Item_Weight"][i] = item_weight_mean.loc[item]["Item_weight"]
      elif miss_bool[i]:
            data["Item_Weight"][i] = np.mean(data["Item_Weight"])
      
# fill the missing values of Outlet_Size Based on the Outlet_Type
  outlet_size_mode = data.pivot_table(values="Outlet_Size", columns ="Outlet_Type", aggfunc =(lambda x: x.mode()[0]))      
        
# now lets fill the missing values
  miss_bool = data["Outlet_Size"].isnull()
  data.loc[miss_bool , "Outlet_Size"] = data.loc[miss_bool , "Outlet_Type"].apply(lambda x: outlet_size_mode[x])    
 
# replace zeros with mean value of Item_visibility
  data.loc[:,"Item_Visibility"].replace([0], [data["Item_Visibility"].mean()], inplace = True)    
    
# LF,low fat = Low Fat ,reg = Regular
  data = data.replace(to_replace = "low fat",value = "Low Fat")
  data = data.replace(to_replace = "LF" , value= "Low Fat")
  data = data.replace(to_replace = "reg" , value = "Regular")   
    
# Creation Of New Attribute
  if "Item_Identifier" in data.columns:
     data["New_Item_Type"] = data["Item_Identifier"].apply(lambda x: x[:2])
     data["New_Item_Type"] = data["New_Item_Type"].replace({"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"})
  else:
     data["New_Item_Type"] = "Unknown"

  data.loc[data["New_Item_Type"] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"

# Create small values for establishent year
  if "Outlet_Establishment_Year" in data.columns:
    data["Outlet_Years"] = 2013 - data["Outlet_Establishment_Year"]
  else:
    data["Outlet_Years"] = 0
  return data

# Streamlit app
def main():
   
   st.title("Item Outlet Sales Prediction App")

# Load Data
   data = load_data()
   print(data.columns)

# Preprocess data
   processed_data = preprocess_data(data)
   print(processed_data.columns)

   X = processed_data[["Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type","Item_MRP",
                       "Outlet_Size","Outlet_Location_Type" ,"Outlet_Type","New_Item_Type","Outlet_Years","Outlet_Identifier"]]
   y = processed_data["Item_Outlet_Sales"]

# Splitting the data into Train and Test
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
   model = train_model(X_train, y_train)
   
# Model Evaluation
   test_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, test_pred)
   st.write("Mean Squared error", mse)
     
# User input for prediction
   st.sidebar.header("Make a Prediction")
   item_weight = st.sidebar.number_input("Item Weight", min_value=0.0, max_value=50.0, step=0.1, value=10.0)
   item_fat_content = st.sidebar.selectbox("Item Fat Content", ["Low Fat Item", "Regular Fat Item", "Non Edible Item"])
   item_visibility = st.sidebar.number_input("Item Visibility", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
   item_type = st.sidebar.selectbox("Item Type", ["Fruits and Vegetables","Snack Foods","Household","Frozen Foods","Dairy",
                                                  "Canned","Baking Goods","Health and Hygiene","Soft Drinks","Meat","Breads",
                                                  "Hard Drinks","Others","Starchy Foods","Breakfast","Seafood""Fruits and Vegetables",
                                                  "Snack Foods","Household","Frozen Foods","Dairy","Canned","Baking Goods",
                                                  "Health and Hygiene","Soft Drinks","Meat","Breads","Hard Drinks",
                                                  "Others","Starchy Foods","Breakfast","Seafood"])
   item_mrp = st.sidebar.number_input("Item MRP", min_value=0.0, max_value=1000.0, step=1.0, value=100.0)
   outlet_identifier = st.sidebar.selectbox("Select The Unique Outlet Number",["10" ,"13","17","18","19",
                                                                               "27","35","45","46","49"])
   outlet_year = st.sidebar.slider("Select The Outlet Age", min_value=1, max_value=20, step=1, value=50)
   outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Midium", "High"])
   outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
   outlet_type = st.sidebar.selectbox("Outlet Type", ["Supermarket Type1","Supermarket Type2","Supermarket Type3","Grocery Store"])
   new_item_type= st.sidebar.selectbox("Types Of Item", ["Food Item", "Drinks Item", "Non-Consumable Item"])

# Make prediction
   input_data = pd.DataFrame({ 'Item_Weight': [item_weight],
                               'Item_Fat_Content': [item_fat_content],
                               'Item_Visibility': [item_visibility],
                               'Item_Type': [item_type],
                               'Item_MRP': [item_mrp],
                               'Outlet_Identifier': [outlet_identifier],
                               'Outlet_Years': [outlet_year],
                               'Outlet_Size': [outlet_size],
                               'Outlet_Location_Type': [outlet_location_type],
                               'Outlet_Type': [outlet_type],
                               'New_Item_Type':[new_item_type]})
     

   input_data = preprocess_data(input_data)

   input_data_encoded = label_encode(input_data, ['Item_Fat_Content', 'Item-Type','New_Item_Type', 'Outlet_Size',
                                                   'Outlet_Location_Type', 'Outlet_Type' ,'Outlet_Identifier'])
     
# Select relevant columns
   selected_columns = ["Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type","Outlet_Identifier", "Item_MRP", "Outlet_Size",
                        "Outlet_Location_Type", "Outlet_Type", "New_Item_Type","Outlet_Years"]
    
   X_input = input_data_encoded[selected_columns]

# Display prediction
   st.subheader("Predicted Item Outlet Sales")
   st.write(model.predict(X_input))
# Run the main function
if __name__ == "__main__":
 main()  


