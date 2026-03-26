import streamlit as st
import requests

# Function to fetch weather data from Open-Meteo
def get_weather_data(latitude, longitude):
    url = f'https://api.open-meteo.com/v1/forecast?latitude={{latitude}}&longitude={{longitude}}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=UTC'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Error fetching data from Open-Meteo')
        return None

# Title of the dashboard
st.title('Weather Dashboard')

# User inputs for location
latitude = st.number_input('Latitude', value=0.0)
longitude = st.number_input('Longitude', value=0.0)

# Fetch and display weather data
weather_data = get_weather_data(latitude, longitude)
if weather_data:
    st.subheader('Current Weather Conditions')
    current_weather = weather_data['current_weather']
    st.write(f"Temperature: {current_weather['temperature']} °C")
    st.write(f"Wind Speed: {current_weather['windspeed']} km/h")
    
    st.subheader('Temperature Trends')
    daily_temps = weather_data['daily']['temperature_2m_max'] + weather_data['daily']['temperature_2m_min']
    st.line_chart(daily_temps)
    
    st.subheader('Precipitation Forecast')
    daily_precip = weather_data['daily']['precipitation_sum']
    st.bar_chart(daily_precip)

# Run the app with `streamlit run weather_dashboard.py`