# Climate-Analysis-Assisted-by-Large-Language-Models

<h2 align="center"><strong>LLM's Climate Analysis - FLASK Application</strong></h2>
<p align="center">
  <span style="font-size: 100px;">🌲👨‍🌾🍀</span>
</p>


### 📜 Description

This is an interactive data visualization and analysis project that allows users to explore climate trends.
It's a Flask-based data visualization application that allows users to upload tabular datasets and generate multiple interactive visualizations based on their questions. The app utilizes LangChain for an agentized Large Language Model (LMM) to analyze the dataset and suggest relevant visualizations.

### 💡 Key Features

- Upload any tabular dataset (CSV format).
- Ask questions about the dataset in natural language.
- Generate multiple visualizations dynamically.
- Uses LangChain, FAISS for vector retrieval, and Google Generative AI for insights.
- Supports multiple visualization types, including histograms, pie charts, correlation heatmaps, and line charts.

---
### 🛠️ How to Install and Run

**📌 Prerequisites**

- Python 3.12.6+

#### 1. Clone the repository
```bash
  git clone https://github.com/SamarKri/LLMs_CLIMATE.git
  cd LLMs_CLIMATE 
```

#### 2. Activate the virtual environment
   ```sh
   python -m venv venv
   venv\Scripts\activate  
   ```

#### 3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

#### 4. Run the application
Start the Flask server:
   ```sh
   python run.py
   ```
#### 5. You can now view your Flask app in your browser.
Open a browser and go to:
   ```sh
   Network URL: http://127.0.0.1:5000/
   ```
---

### 🎯 Technologies Used

- **Flask** - Web framework
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations
- **LangChain & FAISS** - LMM and vector retrieval
- **Google Generative AI** - AI-powered insights

### 🔑 Languages

- Python
- JavaScript
- CSS 
- HTML 

### 🌬️ Future Enhancements
- Deploy on Streamlit or Gradio for an improved UI.
- Expand support for additional dataset formats.
- Improve visualization selection using deeper AI analysis.

### 🌍 Deployment

Git & GitHub: Version control and code sharing.

[Railway](https://railway.com/): For deploying the application online.

```sh
   Open your web browser and go to : https://llmsclimate-production.up.railway.app/
```
---
