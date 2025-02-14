# Estimated Time of Arrival (ETA) Predictor

This project is a **Streamlit-based web application** that predicts the estimated time of arrival (ETA) between two locations based on **distance, traffic, weather conditions, and time of day**.

## 🚀 Features
- 📍 Input **start & end latitude/longitude**
- 🚗 Specify **distance and traffic level**
- 🌦️ Choose **weather condition**
- 📅 Select **day of the week** and **time of day**
- 🤖 **Machine Learning Model** trained using a **Decision Tree Regressor**
- 📊 Displays **predicted ETA** and **model performance (RMSE)**

## 📂 Project Structure
```
📁 ETA-Predictor
│── 📄 app.py                # Main Streamlit application
│── 📄 hyderabad_eta_data.csv # Dataset used for training
│── 📄 README.md             # Project Documentation
```

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ETA-Predictor.git
   cd ETA-Predictor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📜 Requirements
Ensure you have the following libraries installed:
```bash
pandas
scikit-learn
streamlit
numpy
```
You can install them using:
```bash
pip install pandas scikit-learn streamlit numpy
```

## 🧑‍💻 Usage
1. Enter the **route details** including latitude, longitude, and distance.
2. Set the **traffic level, weather condition, day, and hour**.
3. Click **"Predict ETA"** to get the estimated time.

## 🎨 UI Preview
![ETA Predictor UI](https://huggingface.co/spaces/JAGYANSU999/Estimated-Time-of-Arrival-ETA-Predictor)

## 📌 Future Improvements
- 🏎️ Improve model accuracy using **advanced ML models**.
- 📡 Integrate **live traffic & weather APIs**.
- 📍 Add **map visualization**.


---

🔗 **Connect with me:** [LinkedIn](www.linkedin.com/in/jagyansu-padhy-573a1b343) | [GitHub](https://github.com/Jagyansu) | [G-mail](jagyansup@gmail.com)

