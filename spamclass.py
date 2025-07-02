import streamlit as st
import joblib


model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')


st.title("Spam Classifier")


user_input = st.text_area("Enter a message:")


if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        prediction_prob = model.predict_proba(vectorized_input).max()


        if prediction == 'spam':
            st.error(f"Prediction: **SPAM** ({prediction_prob:.2%}confidence)")
        else:
            st.success(f"Prediction: **HAM** ({prediction_prob:.2%}confidence)")

