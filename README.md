# AI-Based Network Intrusion Detection System

This project implements an AI-based Network Intrusion Detection System that identifies
malicious network traffic using Machine Learning techniques. The system analyzes network
packet data and classifies it as normal or suspicious.

The application is built using Python and Streamlit, providing an interactive dashboard
for traffic analysis.

---

## Project Features

- Detects network intrusions using Random Forest algorithm
- Classifies traffic as low-risk or high-risk
- Displays detection accuracy after training
- Simulates analysis of random network packets
- Provides AI-based explanation using Groq API
- Simple and interactive web interface

---

## Dataset Used

- CIC-IDS 2017 (DDoS traffic subset)
- File name: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Contains real network traffic data including benign and attack records

---

## Technologies Used

- Python
- Streamlit
- Machine Learning (Random Forest)
- Pandas and NumPy
- Scikit-learn
- Groq API
- Hugging Face (Deployment)

---

## How to Run the Project

1. Install required libraries:

2. Place the dataset file in the project directory.

3. Run the application:

4. Open the displayed local URL in a browser.

---

## Deployment

The project can be deployed on Hugging Face using Streamlit Spaces.
Upload the following files:
- `app.py`
- `requirements.txt`
- `README.md`
- Dataset file

---

## Output

- Model training accuracy
- Network traffic risk level (Low / High)
- Packet feature values
- AI-based explanation of detected traffic

---

## Conclusion

This project demonstrates how machine learning and AI can be applied to
cybersecurity for detecting network intrusions. It provides a practical
approach to understanding network attack detection.

---

## Author

Student Name: Sinchana G
Project Title: AI-Based Network Intrusion Detection System
