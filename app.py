import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="AI Network Intrusion Detection",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")
st.markdown(
    "This application analyzes network traffic using **Machine Learning** "
    "and highlights suspicious activity with AI-based explanations."
)

# -------------------------------------------------
# DATA FILE
# -------------------------------------------------
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Project Controls")

groq_key = st.sidebar.text_input(
    "Groq API Key (optional)",
    type="password"
)

start_training = st.sidebar.button("Train Intrusion Detection Model")

# -------------------------------------------------
# DATA LOADING FUNCTION
# -------------------------------------------------
@st.cache_data
def load_network_data(file_path):
    df = pd.read_csv(file_path, nrows=15000)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# -------------------------------------------------
# MODEL BUILDING FUNCTION
# -------------------------------------------------
def create_rf_model(data):
    feature_cols = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Fwd Packet Length Max",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow Packets/s"
    ]

    X = data[feature_cols]
    y = data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )

    rf_model = RandomForestClassifier(
        n_estimators=12,
        max_depth=10,
        random_state=7
    )

    rf_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf_model.predict(X_test))

    return rf_model, accuracy, X_test, y_test

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
try:
    network_df = load_network_data(DATA_FILE)
    st.sidebar.success(f"Dataset Loaded: {len(network_df)} records")
except:
    st.error("Dataset file not found. Please upload it to the project folder.")
    st.stop()

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
if start_training:
    with st.spinner("Training model on network traffic..."):
        model, acc, X_test, y_test = create_rf_model(network_df)

        st.session_state["model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success("Model training completed successfully")
        st.info(f"Detection Accuracy Achieved: {round(acc * 100, 2)}%")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
st.header("Network Threat Analysis Dashboard")

if "model" in st.session_state:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Traffic Simulation")
        st.write("Analyze a random packet from the test dataset.")

        if st.button("Analyze Random Network Packet"):
            idx = np.random.randint(0, len(st.session_state["X_test"]))
            packet_sample = st.session_state["X_test"].iloc[idx]
            true_label = st.session_state["y_test"].iloc[idx]

            st.session_state["packet"] = packet_sample
            st.session_state["true_label"] = true_label

    if "packet" in st.session_state:
        packet = st.session_state["packet"]

        with col_left:
            st.write("**Packet Feature Values**")
            st.dataframe(packet.to_frame(name="Value"), use_container_width=True)

        with col_right:
            st.subheader("Detection Outcome")

            prediction = st.session_state["model"].predict([packet])[0]

            if prediction == "BENIGN":
                st.success("Traffic Risk Level: LOW")
                st.write("The packet behavior appears normal and safe.")
            else:
                st.error("Traffic Risk Level: HIGH")
                st.write(f"Suspicious activity detected: {prediction}")

            st.caption(
                f"Reference Dataset Label: {st.session_state['true_label']}"
            )

            st.markdown("---")
            st.subheader("AI-Based Packet Interpretation")

            if st.button("Generate AI Explanation"):
                if not groq_key:
                    st.warning("Please enter a valid Groq API key.")
                else:
                    try:
                        groq_client = Groq(api_key=groq_key)

                        explanation_prompt = f"""
                        You are a cybersecurity analyst.
                        A network packet has been classified as {prediction}.

                        Packet details:
                        {packet.to_string()}

                        Explain in simple terms why this packet
                        is considered {prediction}.
                        """

                        with st.spinner("AI is analyzing packet behavior..."):
                            response = groq_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "user", "content": explanation_prompt}
                                ],
                                temperature=0.6
                            )

                            st.info(response.choices[0].message.content)

                    except Exception as err:
                        st.error(f"Groq API Error: {err}")
else:
    st.info("Please train the model using the sidebar to start analysis.")
