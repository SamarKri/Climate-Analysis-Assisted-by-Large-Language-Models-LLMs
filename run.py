from flask import Flask, render_template, request, redirect, url_for, flash
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in the .env file.")

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "data/uploaded"
PROCESSED_FOLDER = "data/processed"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY)
prompt = PromptTemplate(
    input_variables=["columns", "question"],
    template="Dataset columns: {columns}. User question: {question}. Provide exactly two suggested visualizations in the following structured format:\n\n"
             "1. **[Visualization Type]**\n"
             "* **X-axis:** [X]\n"
             "* **Y-axis:** [Y]\n"
             "* **Additional details:** [Optional]"
)

def run_llm(inputs):
    return llm.invoke(prompt.format(columns=inputs["columns"], question=inputs["question"])).content

llm_chain = RunnableLambda(run_llm)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
            if df.empty:
                flash("The uploaded file is empty.")
                return redirect(request.url)
            
            df = clean_dataset(df)
            df_html = df.head().to_html(classes="table table-striped")  
        except Exception as e:
            flash(f"Error loading file: {str(e)}")
            return redirect(request.url)

        return render_template("upload.html", dataset_name=file.filename, df_html=df_html, columns=df.columns.tolist())
    
    return render_template("upload.html")

def clean_dataset(df):
    """
    Nettoie le dataset en supprimant les valeurs manquantes et les doublons.
    Si la colonne 'temperature_change' existe, elle est convertie en type numérique.
    """
    # Supprimer les doublons et les valeurs manquantes
    df_cleaned = df.dropna().drop_duplicates()
    
    # Si la colonne 'temperature_change' existe, la convertir en numérique
    if 'temperature_change' in df_cleaned.columns:
        df_cleaned['temperature_change'] = pd.to_numeric(df_cleaned['temperature_change'], errors='coerce')
    
    # Supprimer les lignes avec des valeurs manquantes dans 'temperature_change' après la conversion
    if 'temperature_change' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['temperature_change'])
    
    return df_cleaned

@app.route("/process", methods=["POST"])
def process():
    dataset_name = request.form.get("dataset_name")
    question = request.form.get("question")
    selected_column = request.form.get("selected_column", "{}")
    selected_column = json.loads(selected_column) if isinstance(selected_column, str) else {}
    if not dataset_name:
        return redirect(url_for("upload"))
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_name)
    df = pd.read_csv(dataset_path)
    answer_data = answer_question(df, question, selected_column)
    return render_template("results.html", **answer_data)

def answer_question(df, question, selected_column):
    response = run_llm({"columns": ', '.join(df.columns), "question": question})
    formatted_answer = "<ul>\n"
    for line in response.splitlines():
        if line.strip().startswith("*"):
            formatted_answer += f"<li>{line.strip()[1:].strip()}</li>\n"
        else:
            formatted_answer += f"<p>{line.strip()}</p>\n"
    formatted_answer += "</ul>"
    visualization_types = extract_visualization_types(response)
    visualizations = generate_visualizations(df, visualization_types, selected_column)
    return {"answer": formatted_answer, "visualizations": visualizations, "question": question}

def extract_visualization_types(response):
    vis_types = []
    response = response.lower()
    keywords = {
        "correlation": ["correlation", "relationship"],
        "histogram": ["distribution", "histogram"],
        "pie": ["pie", "proportion"],
        "line": ["trend", "time", "line"],
        "bar": ["bar", "comparison"],
        "scatter": ["scatter", "plot"],
        "boxplot": ["boxplot", "distribution", "spread"],
        "violin": ["violin", "distribution", "density"]
    }

    for line in response.splitlines():
        for vis, words in keywords.items():
            if any(word in line for word in words):
                vis_types.append(vis)
                break 
    return vis_types

def generate_visualizations(df, visualization_types, selected_column):
    visualizations = []
    unique_id = str(int(time.time()))

    # Sélection des colonnes par défaut si non spécifiées
    if not selected_column:
        selected_column = {'x': df.columns[0], 'y': df.columns[1] if len(df.columns) > 1 else None}
    
    x_column = selected_column.get('x')
    y_column = selected_column.get('y')

    # Éviter la duplication des visualisations
    vis_types = list(set(visualization_types))  
    
    for vis_type in vis_types:
        if vis_type == "correlation":
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_correlation.png")
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "bar" and y_column:
            fig = px.bar(df, x=x_column, y=y_column, title="Bar Chart")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_bar.html")
            fig.write_html(filepath)
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "line" and y_column:
            fig = px.line(df, x=x_column, y=y_column, title="Line Chart")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_line.html")
            fig.write_html(filepath)
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "histogram":
            fig = px.histogram(df, x=x_column, title="Histogram")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_histogram.html")
            fig.write_html(filepath)
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "pie":
            fig = px.pie(df, names=x_column, title="Pie Chart")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_pie.html")
            fig.write_html(filepath)
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "scatter" and y_column:
            fig = px.scatter(df, x=x_column, y=y_column, title="Scatter Plot")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_scatter.html")
            fig.write_html(filepath)
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "heatmap":
            if x_column in df.columns and y_column in df.columns:
                pivot_table = df.pivot(index=y_column, columns=x_column, values=df.select_dtypes(include=['number']).columns[0])
                plt.figure(figsize=(12, 6))
                sns.heatmap(pivot_table, annot=False, cmap="coolwarm")
                filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_heatmap.png")
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                visualizations.append(os.path.basename(filepath))
            else:
                print("Heatmap visualization skipped due to missing columns.")

        elif vis_type == "boxplot" and y_column:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[x_column], y=df[y_column], palette="Set2")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_boxplot.png")
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            visualizations.append(os.path.basename(filepath))

        elif vis_type == "violin" and y_column:
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=df[x_column], y=df[y_column], palette="Set2")
            filepath = os.path.join(STATIC_FOLDER, f"{unique_id}_violin.png")
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            visualizations.append(os.path.basename(filepath))

    return visualizations

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Railway fournit le port dans l'env
    app.run(host="0.0.0.0", port=port, debug=True)
