# **Travel & Tourism Analytics with Machine Learning and MLOps**

## **Overview**

This repository showcases an end-to-end implementation of multiple machine learning projects in the travel domain, combined with practical **MLOps practices**. The project covers regression, classification, and recommendation systems built using real-world datasets (flights, users, and hotels), and deployed using modern tools such as Flask, Docker, Kubernetes, Airflow, and Streamlit.

The goal is to demonstrate not just model building, but also how to make models **production-ready, scalable, and interactive**.

---

## **Features**

### ✈️ **Flight Price Prediction (Regression)**

* Built a regression model to predict flight prices using historical flight data.
* Performed feature engineering, EDA, and model comparison across multiple algorithms.
* Final model: **Random Forest Regressor** with strong generalization performance.

📌 **Deployment & MLOps:**

* REST API using Flask for real-time predictions
* Containerized using Docker
* Deployed using Kubernetes for scalability
* Automated workflows using Apache Airflow
* Model tracking with MLflow

<img width="940" height="650" alt="image" src="https://github.com/user-attachments/assets/53aa4ca9-a250-4d9e-8115-71d6b23604ef" />

<img width="940" height="364" alt="image" src="https://github.com/user-attachments/assets/ea9517d0-7a0f-4a46-959d-94508f5735ca" />

<img width="940" height="461" alt="image" src="https://github.com/user-attachments/assets/d5e59b7f-79db-40dd-9518-b1e756e158d9" />

---

### 👤 **Gender Classification Model (NLP + Classification)**

* Built a classification model to predict user gender using structured + text data.
* Used **Sentence Transformers** to generate embeddings from names.
* Applied **PCA** for dimensionality reduction.
* Final model: **Tuned Logistic Regression**.

📌 **Deployment:**

* Flask API for serving predictions
* Docker for containerization
* Interactive **Streamlit app** for user input and predictions

<img width="940" height="576" alt="image" src="https://github.com/user-attachments/assets/e75e6d47-d22f-448c-a0b6-35a051da4fcb" />

<img width="741" height="600" alt="image" src="https://github.com/user-attachments/assets/257ec8ce-2fe9-4173-ad58-c383a4053f4f" />
<img width="822" height="317" alt="image" src="https://github.com/user-attachments/assets/27755568-b177-4786-b58b-872e3e5b869e" />

---

### 🏨 **Hotel Recommendation System (Collaborative Filtering)**

* Built a recommendation engine using **Collaborative Filtering (SVD)**.
* Used implicit feedback and user-item interaction matrix.
* Evaluated using **Precision@K and Recall@K**.
* Compared against a popularity-based baseline.

📌 **Deployment:**

* Fully deployed using an interactive **Streamlit web application**
* Users can explore personalized hotel recommendations in real time

<img width="940" height="463" alt="image" src="https://github.com/user-attachments/assets/2048ded7-aeec-4839-8ef8-0586e288f33b" />

---

## **Tech Stack**

### **Languages & Libraries**

* Python 3.10
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Sentence Transformers (NLP)
* Streamlit, Flask

### **MLOps & Deployment Tools**

* Docker
* Kubernetes
* Apache Airflow
* MLflow

---

## **Project Structure**

```
├── data/
│   ├── flights.csv
│   ├── users.csv
│   ├── hotels.csv
├── models/
│   ├── flight_price_model/
│   ├── gender_classification_model/
│   ├── hotel_recommendation_model/
├── api/
│   ├── app.py  (Flask APIs)
├── airflow/
│   ├── dags/
├── deployment/
│   ├── Dockerfile
│   ├── kubernetes.yaml
├── streamlit_app/
│   ├── app.py
├── notebooks/
├── README.md
└── requirements.txt
```

---

## **Setup Instructions**

### **Prerequisites**

* Python 3.10
* Docker installed
* Kubernetes (optional for scaling)
* Apache Airflow setup
* MLflow (for experiment tracking and model management)

---

### **Steps to Run**

#### 1. Clone the Repository

```bash
git clone https://github.com/AimanSahay/MLOps-Project.git
cd MLOps-Project
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### **Run Flask API**

```bash
cd api
python app.py
```

---

### **Run Streamlit App**

```bash
streamlit run streamlit_app/app.py
```

---

### **Docker Setup**

```bash
docker build -t mlops-project .
docker run -p 5000:5000 mlops-project
```

---

### **Kubernetes Deployment (Optional)**

```bash
kubectl apply -f deployment/kubernetes.yaml
```

---

### **Run Airflow DAGs**

* Place DAGs in the Airflow `dags/` folder
* Start Airflow scheduler & webserver

---

### **To start MLflow UI**

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## **Evaluation Metrics**

### **Regression Model**

* RMSE (Root Mean Squared Error)
* R² Score

### **Classification Model**

* Accuracy
* Precision, Recall, F1-score

### **Recommendation System**

* Precision@K
* Recall@K

---

## **Results & Insights**

* Flight Price Prediction achieved strong performance with **R² ≈ 0.86**
* Gender Classification effectively leveraged **NLP embeddings for improved accuracy**
* Hotel Recommendation System provided **personalized recommendations outperforming baseline models**

---

## **Future Enhancements**

* Improve recommendation system with hybrid models
* Add real-time data pipelines
* Enhance UI/UX of Streamlit apps
* Integrate monitoring for deployed models

