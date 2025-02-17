import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Model & Tokenizer
model = tf.keras.models.load_model(r"C:\Users\Smile\data project\lstm_spam_model.h5")
tokenizer = joblib.load(r"C:\Users\Smile\data project\tokenizer.joblib")

# Streamlit App
st.title("📧 AI-Powered Spam Detector")
st.write("Enter a message to check if it's spam or not.")

# User Input
message = st.text_area("Enter your message here:")

# Prediction Function
def predict_spam(message):
    if message:
        seq = tokenizer.texts_to_sequences([message])  # Convert text to sequence
        pad_seq = pad_sequences(seq, maxlen=100)  # Pad sequence
        prediction = model.predict(pad_seq)[0][0]  # Get prediction
        if prediction > 0.4:
           return "🛑 Spam", "red"
        else:
            return "✅ Not Spam", "green"

# Button for Prediction
if st.button("Check Message"):
    result, color = predict_spam(message)
    
    # Display Result with Color
    st.markdown(f"<h2 style='color: {color};'>{result}</h2>", unsafe_allow_html=True)