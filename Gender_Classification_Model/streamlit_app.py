import streamlit as st
import requests

# Flask API URL
API_URL = "http://localhost:8000/predict"

def run():
    st.title("Gender Classification Model")

    # Inputs
    usercode = st.number_input("Usercode", min_value=0, max_value=1339, value=1234)
    company = st.selectbox("Company Name", ["Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA", "4You"])
    name = st.text_input("Username", "Charlotte Johnson")
    age = st.slider("Traveller Age", 21, 65, 30)

    if st.button("Predict"):

        data = {
            "Usercode": usercode,
            "company_name": company,
            "Username": name,
            "Traveller_Age": age
        }

        try:
            response = requests.post(API_URL, data=data)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Gender: {result['prediction']}")
            else:
                st.error("Error in prediction API")

        except Exception as e:
            st.error(f"Connection error: {e}")


if __name__ == "__main__":
    run()