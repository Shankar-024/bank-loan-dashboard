from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

DATA_PATH = "loan_data.csv"
MODEL_PATH = "model.pkl"
PLOTS_DIR = "static/plots"
REPORT_PATH = "static/report.txt"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- GLOBAL STATE ----------------
model = None
training_status = "🧪 Demo model (upload CSV to retrain)"

# ---------------- DEMO MODEL (FIXES HIGH RISK BUG) ----------------
def load_demo_model():
    global model

    X_demo = np.array([
        [25, 30000, 50000, 36, 750, 0],
        [45, 12000, 200000, 60, 450, 3]
    ])
    y_demo = np.array([0, 1])  # BOTH CLASSES (IMPORTANT)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    model.fit(X_demo, y_demo)

load_demo_model()

# ---------------- EDA ----------------
def generate_eda(df):
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png")
    plt.clf()

    df["credit_score"].hist()
    plt.title("Credit Score Distribution")
    plt.xlabel("Credit Score")
    plt.ylabel("Count")
    plt.savefig(f"{PLOTS_DIR}/credit_score_dist.png")
    plt.clf()

    plt.scatter(df["loan_amount"], df["credit_score"])
    plt.xlabel("Loan Amount")
    plt.ylabel("Credit Score")
    plt.title("Loan Amount vs Credit Score")
    plt.savefig(f"{PLOTS_DIR}/loan_vs_credit.png")
    plt.clf()

    df.groupby("default")["income"].mean().plot(kind="bar")
    plt.title("Income vs Default")
    plt.savefig(f"{PLOTS_DIR}/income_vs_default.png")
    plt.clf()

    df["default"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Default Distribution")
    plt.savefig(f"{PLOTS_DIR}/default_pie.png")
    plt.clf()

# ---------------- METRICS ----------------
def get_model_metrics():
    if not os.path.exists(DATA_PATH):
        return "N/A", "N/A", "Upload dataset to see metrics"

    df = pd.read_csv(DATA_PATH)
    X = df[['age','income','loan_amount','loan_term_months','credit_score','existing_emis']]
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    acc = round(accuracy_score(y_test, preds)*100, 2)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    cm_html = "<table>"
    for r in cm:
        cm_html += "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
    cm_html += "</table>"

    return acc, cm_html, report

# ---------------- HOME ----------------
@app.route("/")
def home():

    # Default empty values (before upload)
    total_records = "-"
    avg_income = "-"
    avg_credit = "-"
    default_rate = "-"
    accuracy = "-"
    training_status = "⏳ Waiting for dataset upload"

    # If CSV exists → compute values
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        total_records = len(df)
        avg_income = round(df["income"].mean(), 2)
        avg_credit = round(df["credit_score"].mean(), 2)
        default_rate = f"{round(df['default'].mean()*100, 2)}%"

        accuracy, cm_html, report = get_model_metrics()
        training_status = "✅ Model Trained"

        return render_template(
            "index.html",
            total_records=total_records,
            avg_income=avg_income,
            avg_credit=avg_credit,
            default_rate=default_rate,
            accuracy=accuracy,
            training_status=training_status,
            confusion_matrix_html=cm_html,
            classification_report=report,
            dataset_summary=df.describe().to_html(classes="summary-table")
        )

    # First load (no CSV yet)
    return render_template(
        "index.html",
        total_records=total_records,
        avg_income=avg_income,
        avg_credit=avg_credit,
        default_rate=default_rate,
        accuracy=accuracy,
        training_status=training_status
    )

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    values = [[
        float(request.form["age"]),
        float(request.form["income"]),
        float(request.form["loan_amount"]),
        float(request.form["loan_term_months"]),
        float(request.form["credit_score"]),
        float(request.form["existing_emis"])
    ]]

    input_df = pd.DataFrame(values, columns=[
        'age','income','loan_amount',
        'loan_term_months','credit_score','existing_emis'
    ])

    proba = model.predict_proba(input_df)[0][1]

    if proba >= 0.7:
        text, color = "⚠️ High Risk", "red"
    elif proba >= 0.4:
        text, color = "🟠 Medium Risk", "orange"
    else:
        text, color = "✅ Low Risk", "green"

    acc, cm, report = get_model_metrics()

    return render_template(
        "index.html",
        prediction_text=text,
        confidence=round(proba*100, 2),
        risk_color=color,
        accuracy=acc,
        confusion_matrix_html=cm,
        classification_report=report,
        training_status=training_status
    )

# ---------------- UPLOAD & TRAIN ----------------
@app.route("/upload", methods=["POST"])
def upload():
    global model, training_status

    file = request.files["file"]
    file.save(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df[['age','income','loan_amount','loan_term_months','credit_score','existing_emis']]
    y = df['default']

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)

    generate_eda(df)

    training_status = "✅ Model retrained using uploaded dataset"

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return redirect(url_for("home"))

# ---------------- REPORT ----------------
@app.route("/download-report")
def download_report():
    acc, _, report = get_model_metrics()
    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {acc}%\n\n{report}")
    return send_file(REPORT_PATH, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
