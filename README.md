# Data Assistant Pro 🚀

An enterprise-grade, end-to-end Streamlit application for automated data cleaning, exploratory data analysis (EDA), and machine learning.

![App Screenshot](/Users/ibrahimfofanah/Desktop/Data Assistant/app.png) <!-- You can upload a screenshot to your repo and link it here -->

---

## ✨ Key Features

-   **🤖 Automated EDA:** Generates a comprehensive and interactive dashboard analyzing data types, missing values, distributions, and correlations.
-   **🧹 Advanced Data Cleaning:** Interactive UI for handling missing values, removing duplicates, and optimizing data types for memory efficiency.
-   **🤖 Enterprise AutoML:** Automatically trains and compares multiple machine learning models (Random Forest, Linear/Logistic Regression, SVM) for both classification and regression tasks.
-   **📊 Rich Visualizations:** Uses Plotly for interactive charts, including model performance comparisons, prediction plots, and feature importance analysis.
-   **📈 Reporting & Export:** Generates a data cleaning report with a "Data Quality Score" and allows exporting the cleaned dataset to CSV or Excel.
-   **🛡️ Robust & User-Friendly:** Features comprehensive error handling, data validation, a professional UI, and a built-in sample dataset for immediate use.

---

## 🛠️ Tech Stack

-   **Language:** Python
-   **Framework:** Streamlit
-   **Data Manipulation:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn
-   **Visualization:** Plotly
-   **Logging & Configuration**

---

## ⚙️ Setup and Installation

To run this application locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Denis060/data-assistant-pro.git](https://github.com/Denis060/data-assistant-pro.git)
    cd data-assistant-pro
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run

Once the setup is complete, you can launch the application with the following command:

```bash
streamlit run app.py