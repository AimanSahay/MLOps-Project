from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# =========================
# Load Models
# =========================
embedding_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

scaler = pickle.load(open("models/scaler.pkl", "rb"))
pca = pickle.load(open("models/pca.pkl", "rb"))
model = pickle.load(open("models/tuned_logistic_regression_model.pkl", "rb"))
ohe = pickle.load(open("models/ohe.pkl", "rb"))


# =========================
# Prediction Function
# =========================
def predict_gender(input_data):

    df = pd.DataFrame([input_data])

    # =========================
    # Data Cleaning
    # =========================
    df['code'] = df['code'].astype(int)
    df['age'] = df['age'].astype(int)
    df['name'] = df['name'].fillna('').astype(str)

    # =========================
    # OneHot Encoding
    # =========================
    company_ohe = ohe.transform(df[['company']]).toarray() 

    # =========================
    # Text Embeddings
    # =========================
    embeddings = embedding_model.encode(df['name'].tolist())

    # =========================
    # PCA Transform (NOT fit!)
    # =========================
    embeddings_pca = pca.transform(embeddings)

    # =========================
    # Numerical Features
    # =========================
    X_numerical = np.hstack((
        df[['code', 'age']].values.reshape(1, -1),
        company_ohe
    ))

    print("embeddings_pca shape:", embeddings_pca.shape)
    print("X_numerical shape:", X_numerical.shape)

    # =========================
    # Combine Features
    # =========================
    X = np.hstack((embeddings_pca, X_numerical))

    # =========================
    # Scaling
    # =========================
    X = scaler.transform(X)

    # =========================
    # Prediction
    # =========================
    pred = model.predict(X)[0]

    return int(pred)


# =========================
# Routes
# =========================
@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
    <h2>Gender Classification</h2>
    <form action="/predict" method="POST">
        Name: <input type="text" name="Username" value="Charlotte Johnson"><br><br>
        Code: <input type="number" name="Usercode"><br><br>
        Age: <input type="number" name="Traveller_Age"><br><br>

        Company:
        <select name="company_name">
            <option value="Acme Factory">Acme Factory</option>
            <option value="Wonka Company">Wonka Company</option>
            <option value="Monsters CYA">Monsters CYA</option>
            <option value="Umbrella LTDA">Umbrella LTDA</option>
            <option value="4You">4You</option>
        </select><br><br>

        <input type="submit" value="Predict">
    </form>
    """)


@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = {
            'code': request.form.get('Usercode'),
            'company': request.form.get('company_name'),
            'name': request.form.get('Username'),
            'age': request.form.get('Traveller_Age'),
        }

        pred = predict_gender(data)

        gender_map = {
            0: "female",
            1: "male"
        }

        result = gender_map.get(pred, "unknown")

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)