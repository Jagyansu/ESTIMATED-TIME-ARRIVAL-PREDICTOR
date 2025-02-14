import pandas as pd
import sklearn
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_selection import VarianceThreshold

df=pd.read_csv("C:/Users/Mr.Jagyansu/Downloads/ETA/hyderabad_eta_data.csv")
st.header(":green[Estimated Time of Arrival (ETA) Predictor]")
st.subheader("by Jagyansu Padhy")

st.markdown("""
This tool helps you estimate the travel time between two locations based on **distance, traffic, weather conditions, and time of day**.  Simply **select your route details**, and the model will predict the estimated time of arrival (ETA) in minutes.
""")
weather_map = {'rainy': 0, 'clear': 1, 'foggy': 2}
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

df['weather_condition'] = df['weather_condition'].map(weather_map)
df['day_of_week'] = df['day_of_week'].map(day_map)


X=df.drop('ETA',axis=1)
y=df['ETA']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


selector = VarianceThreshold(threshold=0)
X_train_selected = selector.fit_transform(X_train_imputed)
X_test_selected = selector.transform(X_test_imputed)

if X_train_selected.shape[1] == 0:
    st.error("All features were removed due to zero variance. Try adjusting preprocessing!")
    st.stop()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

@st.cache_resource
def train_model():
    dt = DecisionTreeRegressor()
    dt.fit(X_train_scaled, y_train)
    return dt
dt_model = train_model()

st.subheader("Enter Route Details:")


col1, col2 = st.columns(2)

with col1:
    start_lat = st.number_input("Start Latitude", min_value=-90.0, max_value=90.0, value=40.7128, step=1.00, format="%.2f")
    start_lng = st.number_input("Start Longitude", min_value=-180.0, max_value=180.0, value=-74.0060, step=1.00, format="%.2f")
    end_lat = st.number_input("End Latitude", min_value=-90.0, max_value=90.0, value=34.0522, step=1.00, format="%.2f")
    end_lng = st.number_input("End Longitude", min_value=-180.0, max_value=180.0, value=-118.2437, step=1.00, format="%.2f")

with col2:
    distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=5000.0, value=100.0, step=0.1)
    def get_traffic_color(level):
        colors = ["#00FF00", "#33FF00", "#66FF00", "#99FF00", "#CCFF00", 
              "#FFFF00", "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000"]
        return colors[level]

traffic_density = st.slider(" 🚦Traffic Level (0-10)", min_value=0, max_value=10, value=5)

st.markdown(
    f'<div style="background-color:{get_traffic_color(traffic_density)}; color:black; padding:10px; text-align:center; border-radius:12px;">'
    f'<b>Traffic Level: {traffic_density}</b></div>', 
    unsafe_allow_html=True
)


weather_options = {
    "☀️ Clear": ("clear", "#FFD700"),  # Gold color for clear weather
    "🌧️ Rainy": ("rainy", "#00BFFF"),  # Deep Sky Blue for rain
    "🌫️ Foggy": ("foggy", "#A9A9A9")   # Dark Gray for fog
}

# Creating Stylish Weather Buttons
selected_weather = st.radio(
    "🌦️Choose Weather Condition:",
    list(weather_options.keys()),
    horizontal=True
)

# Extract the selected weather type and corresponding color
weather_condition, bg_color = weather_options[selected_weather]

# Styled Weather Display with Dynamic Background
st.markdown(f"""
    <style>
        .weather-container {{
            background-color: {bg_color};
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: black;
            transition: background-color 0.3s ease;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }}
    </style>
    <div class="weather-container">
        {selected_weather} Selected
    </div>
""", unsafe_allow_html=True)


import streamlit as st

# 📅 Mapping Days with Icons and Colors
day_options = {
    "🌞 Monday": ("Monday", "#FFDD44"),
    "🚀 Tuesday": ("Tuesday", "#FF8844"),
    "🌿 Wednesday": ("Wednesday", "#44BB66"),
    "🔥 Thursday": ("Thursday", "#FF4444"),
    "🎉 Friday": ("Friday", "#6633FF"),
    "🛌 Saturday": ("Saturday", "#AA66CC"),
    "☀️ Sunday": ("Sunday", "#00AEEF"),
}



# Stylish Day Selector
selected_day = st.radio(
    "📅 Select the Day of the Week:",
    list(day_options.keys()),
    horizontal=True
)

# Extract selected day and color
day_of_week, day_color = day_options[selected_day]

# Dynamic Styled Display
st.markdown(f"""
    <style>
        .day-container {{
            background-color: {day_color};
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: white;
            transition: background-color 0.3s ease;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        }}
    </style>
    <div class="day-container">
        {selected_day} Selected
    </div>
""", unsafe_allow_html=True)




# Hour Slider with Color Change
hour_of_day = st.slider(
    "⏰Choose the Hour (24-hour format)",
    0, 23, 5, step=1,
    help="Select the time of day (24-hour format)"
)

# Apply color based on hour of the day

#weather_map={'rainy':0,'clear':1,'foggy':2}
#day_map={"Monday":0, "Tuesday":1, "Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

input=pd.DataFrame({'start_lat': [start_lat],
                    'start_lng': [start_lng],
                    'end_lat': [end_lat],
                    'end_lng' : [end_lng],
                    'distance_km' : [distance_km],
                    'traffic_density' : [traffic_density],
                    'weather_condition' : [weather_map[weather_condition]],
                    'day_of_week': [day_map[day_of_week]],
                    'hour_of_day' : [hour_of_day]})

st.markdown("""
    <div style="background-color:#eeeeee; padding:15px; border-radius:12px; font-size:18px; text-align:center;">
        📍 <b>Start Location:</b> Latitude = {0}, Longitude = {1} <br>
        📍 <b>End Location:</b> Latitude = {2}, Longitude = {3} <br>
        🚗 <b>Distance:</b> {4} km <br>
        🚦 <b>Traffic Level:</b> {5} <br>
        🌦️ <b>Weather:</b> {6} <br>
        📅 <b>Day of the Week:</b> {7} <br>
        ⏰ <b>Hour of the Day:</b> {8}
    </div>
""".format(start_lat, start_lng, end_lat, end_lng, distance_km, traffic_density, weather_condition, day_of_week, hour_of_day),
unsafe_allow_html=True)

# Handle missing values in the user input
input_imputed = imputer.transform(input)
input_selected=selector.transform(input_imputed)
input_scaled = scaler.transform(input_selected)

#st.write("**User Input Before Scaling:**")
#st.write(input)
#st.write("**User Input After Scaling:**",input_scaled)

import streamlit as st

# 🌟 ETA Display with Styling
def display_eta(eta_value):
    st.markdown(f"""
        <div style="background-color:white; padding:20px; border-radius:15px; font-size:20px; font-weight:bold; text-align:center; color:#00AEEF; box-shadow:0px 4px 10px rgba(0,0,0,0.2); transition: background-color 0.3s ease;">
            🚗 The Estimated Time of Arrival (ETA): <br>
            <span style="font-size:20px;">{eta_value:.2f} minutes</span>
        </div>
    """, unsafe_allow_html=True)

# 🌟 Model Performance Display with Styling
def display_model_performance(rmse_value):
    st.markdown(f"""
        <div style="background-color:#00AEEF; padding:10px; border-radius:15px; font-size:20px; font-weight:bold; text-align:center; color:white; box-shadow:0px 4px 10px rgba(0,0,0,0.2); transition: background-color 0.3s ease;">
            📊 Model Performance: <br>
            <span style="font-size:20px;">RMSE: {rmse_value:.2f} minutes</span>
        </div>
    """, unsafe_allow_html=True)

# **Interactive Button Style**
st.markdown("""
    <style>
        .stButton>button {
            background-color: #44BB66;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 12px 30px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #44BB66;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# **Prediction & Performance**
if st.button("Click to Predict ETA"):
    # Assuming y_pred[0] and a are the predicted ETA and RMSE values
    y_pred = 25.53  # Example ETA value
    a = 8.72  # Example RMSE value

    display_eta(y_pred)
    display_model_performance(a)