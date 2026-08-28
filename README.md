<div align="center">

# 🏦 [ADD-YOUR-PROJECT-NAME-HERE]
### Customer Credit Risk — End-to-End Data Preprocessing & Feature Engineering

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=260&section=header&text=Customer%20Credit%20Risk%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=End-to-End%20Data%20Preprocessing%20%26%20Feature%20Engineering%20Pipeline&descAlignY=55&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=600&size=22&duration=2500&pause=900&color=00F7FF&center=true&vCenter=true&width=800&lines=Raw+Data+%E2%9E%9C+Cleaned+Data+%E2%9E%9C+ML-Ready+Dataset;500+Rows+%C3%97+15+Columns+%E2%9E%9C+500+Rows+%C3%97+27+Columns;Missing+Values%3A+0+%7C+Duplicates%3A+0+%7C+Target%3A+default_flag;Built+with+Pandas+%2B+NumPy+%2B+Scikit-learn" alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/Scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
<img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib"/>
<img src="https://img.shields.io/badge/Flask-Dummy%20API-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
<img src="https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge" alt="Completed"/>

<br/>

<img src="https://img.shields.io/badge/Rows-500-6C5CE7?style=for-the-badge" alt="Rows"/>
<img src="https://img.shields.io/badge/Columns-15%20%E2%86%92%2027-FD79A8?style=for-the-badge" alt="Columns"/>
<img src="https://img.shields.io/badge/Missing%20Values-0-00B894?style=for-the-badge" alt="Missing"/>
<img src="https://img.shields.io/badge/Duplicates-0-00B894?style=for-the-badge" alt="Duplicates"/>
<img src="https://img.shields.io/badge/Target-default__flag-FDCB6E?style=for-the-badge" alt="Target"/>

<br/><br/>

**A complete academic Data Preprocessing + Feature Engineering project**
**based on a Customer Credit Risk dataset — from raw multi-source data to a fully ML-ready dataset.**

</div>

<br/>

## 📌 Table of Contents

| | | |
|---|---|---|
| [✨ Overview](#-project-overview) | [🧠 Mind Map](#-pipeline-mind-map) | [🎯 Problem Statement](#-problem-statement) |
| [📊 Dataset](#-dataset) | [📥 Data Acquisition](#-data-acquisition) | [🔎 Data Understanding](#-data-understanding) |
| [🧹 Cleaning & Missing Values](#-data-cleaning--missing-values) | [📉 Outlier Handling](#-outlier-handling) | [🔤 Encoding](#-feature-engineering--encoding) |
| [📦 Binning & Binarization](#-binning--binarization) | [📏 Feature Scaling](#-feature-scaling) | [🔄 Transformations](#-feature-transformations) |
| [🧩 ColumnTransformer](#-columntransformer) | [🧠 Constructed Features](#-constructed-features) | [📈 Column Growth](#-column-growth-breakdown) |
| [💾 Final Dataset](#-final-dataset) | [🗂️ Project Structure](#️-project-structure) | [🖼️ Visualizations](#️-visualizations) |
| [🚀 How to Run](#-how-to-run) | [📚 Techniques Covered](#-techniques-covered) | [🎓 Key Learnings](#-key-learnings) |
| [✅ Final Outcome](#-final-outcome) | [👨‍💻 Author](#-author) | [⭐ Support](#-support-this-project) |

<br/>

## ✨ Project Overview

This project demonstrates a **complete, end-to-end workflow** for preparing customer credit-risk data for Machine Learning.

The pipeline moves data through **12 structured stages** — from raw multi-source ingestion, through cleaning, imputation, outlier treatment, encoding, binning, scaling and transformation, all the way to engineered feature construction and a final ML-ready dataset.

> 🎯 **Project Objective:** Transform a raw **500 × 15** customer credit-risk dataset into a fully processed, **500 × 27** ML-ready dataset suitable for downstream modeling.

<br/>

## 🧠 Pipeline Mind Map

<div align="center">

```mermaid
mindmap
  root((Credit Risk ML Pipeline))
    Data Acquisition
      CSV
      JSON
      SQL
      Dummy API
    Data Understanding
      info and describe
      Target Analysis
      Dtype Audit
    Cleaning and Imputation
      Mean / Median
      Most Frequent
      KNN Imputer
      MICE
      Complete Case
    Outlier Treatment
      Z-Score
      IQR
      Percentile
      Winsorization
    Encoding
      Ordinal
      Label / Binary
      One-Hot
    Binning
      Quantile Binning
      K-Means Binning
      Binarization
    Scaling
      Standardization
      Min-Max
      MaxAbs
      Robust
    Transformations
      Log / Sqrt / Reciprocal
      Box-Cox
      Yeo-Johnson
    Feature Construction
      Debt-to-Income
      Avg Monthly Txns
      Spend-to-Income
    Final Dataset
      500 rows
      27 columns
      ML Ready
```

</div>

### 🎬 Pipeline Flow — Step-by-Step (Vertical View)

```mermaid
flowchart TD
    A["📥 Data Acquisition<br/>CSV + JSON + SQL + API merged<br/>500 rows × 15 cols"] --> B["🔎 Data Understanding<br/>dtype audit • missing scan • target check<br/>500 × 15"]
    B --> C["🧹 Data Cleaning<br/>Duplicate records removed<br/>500 × 15"]
    C --> D["🩹 Missing Value Treatment<br/>income, credit_score, repayment history imputed<br/>Missing → 0"]
    D --> E["📉 Outlier Handling<br/>Z-Score / IQR / Winsorization on income & loan_amount<br/>500 × 15"]
    E --> F["🔤 Encoding<br/>join_date split +3 • region one-hot +3 • loan_purpose one-hot +2<br/>15 → 23 cols"]
    F --> G["📦 Binning & Binarization<br/>income_bin +1 • high_credit_score +1<br/>23 → 25 cols"]
    G --> H["📏 Feature Scaling<br/>Standardization / MinMax / Robust on numeric fields<br/>500 × 25"]
    H --> I["🔄 Transformations<br/>Log / Box-Cox / Yeo-Johnson on skewed features<br/>500 × 25"]
    I --> J["🧩 Feature Construction<br/>debt_to_income_ratio +1 • avg_monthly_transactions +1<br/>25 → 27 cols"]
    J --> K["🛠️ ColumnTransformer<br/>Numeric + categorical pipelines unified<br/>500 × 27"]
    K --> L["💾 Final ML-Ready Dataset<br/>0 missing • 0 duplicates • default_flag preserved<br/>✅ 500 × 27"]

    style A fill:#FF6B6B,color:#fff,stroke:#333,stroke-width:1px
    style E fill:#FDCB6E,color:#000
    style F fill:#74B9FF,color:#000
    style H fill:#A29BFE,color:#000
    style L fill:#00B894,color:#fff,stroke:#333,stroke-width:1px
```

### 📋 Stage-by-Stage Interpretation (This Dataset)

| Stage | What Happens to *This* Dataset | Rows | Cols |
|---|---|---|---|
| 📥 Data Acquisition | CSV + JSON + SQL + Dummy API merged into one working dataframe | 500 | 15 |
| 🔎 Data Understanding | dtype audit, missing-value scan, `default_flag` balance check | 500 | 15 |
| 🧹 Data Cleaning | Duplicate customer records identified & removed | 500 | 15 |
| 🩹 Missing Value Treatment | Nulls in `annual_income`, `credit_score`, `repayment_history` imputed (Mean/Median/KNN/MICE) | 500 | 15 |
| 📉 Outlier Handling | Extreme values in `annual_income` & `loan_amount` treated (Z-Score/IQR/Winsorization) | 500 | 15 |
| 🔤 Encoding | `join_date` split (+3), `region` one-hot (+3), `loan_purpose` one-hot (+2); `education_level`/`gender` encoded in place | 500 | **23** |
| 📦 Binning & Binarization | `income_bin` and `high_credit_score` flags added | 500 | **25** |
| 📏 Feature Scaling | Standardization/MinMax/Robust applied to numeric columns (no column change) | 500 | 25 |
| 🔄 Transformations | Log/Box-Cox/Yeo-Johnson applied to skewed numeric features (no column change) | 500 | 25 |
| 🧩 Feature Construction | `debt_to_income_ratio` & `avg_monthly_transactions` added | 500 | **27** |
| 🛠️ ColumnTransformer | Numeric & categorical pipelines unified into one transformer | 500 | 27 |
| 💾 Final Dataset | `Missing = 0`, `Duplicates = 0`, ML-ready | 500 | **27** |

<br/>

## 🎯 Problem Statement

A fintech company provides customer credit-risk data collected from **multiple sources**. The goal is to prepare the data so a Machine Learning model can predict:

> **Will a customer default on a loan?**

| Target Variable | Value | Meaning |
|---|---|---|
| `default_flag` | `0` | No Default |
| `default_flag` | `1` | Default |

**Problem Type:** Binary Classification

<br/>

## 📊 Dataset

The **raw** Customer Credit Risk dataset contains:

**👤 Demographics** — Age • Gender • Region • Education Level • Employment Type
**💰 Financial Details** — Annual Income • Loan Amount • Loan Purpose • Credit Score
**📈 Behavioural Attributes** — Repayment History • Transaction Count • Spending Ratio
**🪪 Identifiers & Dates** — Customer ID • Join Date
**🎯 Target** — Default Flag

### 📐 Dataset Shape — Before vs After

| Stage | Rows | Columns | Notes |
|---|---|---|---|
| 📥 Raw Dataset (multi-source merge) | 500 | **15** | Original demographic + financial + behavioural fields |
| ⚙️ After Cleaning, Encoding, Binning & Construction | 500 | growing… | Encoding & feature construction add new columns |
| 💾 **Final ML-Ready Dataset** | 500 | **27** | Missing values = 0, Duplicates = 0 |

```
Raw Dataset            Final Dataset
┌─────────────┐        ┌─────────────┐
│  500 × 15   │  ───▶  │  500 × 27   │
└─────────────┘        └─────────────┘
   14 processing stages of cleaning, encoding & engineering
```

<br/>

## 📥 Data Acquisition

The project demonstrates data acquisition from **four different source types**, merged into a single working dataframe:

```mermaid
flowchart TD
    A[CSV] -->|Main customer / transaction data| E[Merged Dataframe]
    B[JSON] -->|Customer metadata| E
    C[SQL] -->|Loan repayment history| E
    D[Dummy API] -->|External economic indicators| E
    E --> F[500 × 15 Raw Dataset]

    style E fill:#6C5CE7,color:#fff
    style F fill:#00B894,color:#fff
```

- **CSV** — primary customer & transaction records
- **JSON** — supplementary customer metadata
- **SQL** — loan repayment history table
- **Dummy API** — locally simulated external economic indicators

<br/>

## 🔎 Data Understanding

The dataset was explored using Pandas:

```python
df.info()
df.describe()
df.head()
```

Focus areas:
- Number of rows and columns (500 × 15)
- Data types (numerical vs categorical vs date)
- Missing values per column
- Basic statistical properties (mean, std, quartiles)
- Target variable balance (`default_flag`)

<br/>

## 🧹 Data Cleaning & Missing Values

| Technique | Purpose |
|---|---|
| Simple Imputer — Mean | Numerical missing values |
| Simple Imputer — Median | Numerical missing values (skewed data) |
| Simple Imputer — Most Frequent | Categorical missing values |
| Most Frequent Category Imputation | Categorical data |
| Missing Indicator + Random Sample | Preserve missingness information |
| KNN Imputer | Multivariate numerical imputation |
| MICE | Iterative multivariate imputation |
| Complete Case Analysis | Removing incomplete observations |

**Final Cleaning Result**

```
Missing values before  → Present across multiple columns
Missing values after   → 0
Duplicate rows         → 0
```

<br/>

## 📉 Outlier Handling

Four approaches were explored and compared:

| Method | Core Idea |
|---|---|
| **Z-Score** | Flag observations where `\|Z\| > 3` |
| **IQR** | `IQR = Q3 − Q1`, bounds = `Q1 − 1.5×IQR` / `Q3 + 1.5×IQR` |
| **Percentile** | Extreme values identified via percentile boundaries |
| **Winsorization** | Extreme values capped rather than deleted — preserves records |

```mermaid
flowchart TD
    A[Numerical Feature] --> B[Detect Extreme Values]
    B --> C{Method}
    C --> D[Z-Score]
    C --> E[IQR]
    C --> F[Percentile]
    C --> G[Winsorization]
    D --> H[Compare Before / After]
    E --> H
    F --> H
    G --> H
    H --> I[Cleaned Feature]

    style H fill:#FDCB6E,color:#000
    style I fill:#00B894,color:#fff
```

<br/>

## 🔤 Feature Engineering & Encoding

### 🏷️ Mixed Variable Types
The dataset contains **numerical**, **categorical**, and **date** variables — each handled with an appropriate technique.

### 📅 Date & Time Features
`join_date` was decomposed into:

```
join_date  ──▶  join_year
            ──▶  join_month
            ──▶  join_day
            ──▶  join_weekday
```

### 🔢 Ordinal Encoding
Applied to `education_level` because categories have a natural order:

```
Primary → Secondary → Graduate → Post-Graduate
```

### 🏷️ Label / Binary Encoding
Binary-nature features (e.g. `gender`, `employment_type`) represented numerically as `0` / `1`.

### 🧱 One-Hot Encoding
Applied to nominal categorical variables:

```
region        → region_North, region_South, region_East, region_West
loan_purpose  → loan_purpose_Education, loan_purpose_Business, loan_purpose_Personal
```

Creates separate binary columns without imposing an artificial ranking.

<br/>

## 📦 Binning & Binarization

| Technique | Description |
|---|---|
| **Income Binning** | `Low` / `Medium` / `High` / `Very High` categories |
| **Quantile Binning** | Income divided into quantile-based groups |
| **K-Means Binning** | Income grouped using K-Means clustering |
| **Binarization** | `credit_score > 700 → high_credit_score = 1`, else `0` |

<br/>

## 📏 Feature Scaling

| Method | Main Idea |
|---|---|
| **Standardization** | Mean ≈ 0, Std ≈ 1 |
| **Normalization** | Scale observations to a common magnitude |
| **Min-Max Scaling** | Maps values to a `0–1` range |
| **MaxAbs Scaling** | Scales using maximum absolute value |
| **Robust Scaling** | Uses median and IQR — resistant to outliers |

**Why scale?**

```
Income        → hundreds of thousands
Loan Amount   → hundreds of thousands
Credit Score  → hundreds
```

Scaling makes numerical features comparable for magnitude-sensitive algorithms.

<br/>

## 🔄 Feature Transformations

**FunctionTransformer**

```python
from sklearn.preprocessing import FunctionTransformer

log_transform = FunctionTransformer(func=np.log1p)
sqrt_transform = FunctionTransformer(func=np.sqrt)
reciprocal_transform = FunctionTransformer(func=lambda x: 1 / (x + 1))
```

- Log Transform
- Reciprocal Transform
- Square Root Transform

**PowerTransformer**

```python
from sklearn.preprocessing import PowerTransformer

boxcox = PowerTransformer(method="box-cox")        # positive values only
yeojohnson = PowerTransformer(method="yeo-johnson") # handles zero & negative values
```

| Method | Requirement |
|---|---|
| **Box-Cox** | Requires strictly positive values |
| **Yeo-Johnson** | Handles zero and negative values |

<br/>

## 🧩 ColumnTransformer

Different preprocessing operations applied to different columns in a single unified workflow:

```mermaid
flowchart LR
    A[Dataset] --> B{Column Type}
    B --> C[Numerical]
    B --> D[Categorical]
    C --> E[Scaling / Transformation]
    D --> F[Encoding]
    E --> G[Combined Processed Dataset]
    F --> G

    style G fill:#00B894,color:#fff
```

This keeps the preprocessing pipeline organized and production-ready for ML pipelines (`Pipeline` + `ColumnTransformer`).

<br/>

## 🧠 Constructed Features

| # | Feature | Formula | Meaning |
|---|---|---|---|
| 1 | **Debt-to-Income Ratio** | `loan_amount / annual_income` | Loan burden relative to income |
| 2 | **Average Monthly Transactions** | `transaction_count / 6` | Monthly average (6-month window) |
| 3 | **Spending-to-Income Ratio** | `spending_ratio` (retained) | Existing engineered spend/income signal |

<br/>

## 📈 Column Growth Breakdown

> A sample breakdown of how the dataset grows from **15 → 27** columns across the pipeline. Adjust the exact feature names below to match your own notebook.

| Stage | Change | Running Total |
|---|---|---|
| Raw dataset | — | **15** |
| `join_date` → year/month/day/weekday (drop original) | −1, +4 | 18 |
| `region` → one-hot (4 dummies, drop original) | −1, +4 | 21 |
| `loan_purpose` → one-hot (3 dummies, drop original) | −1, +3 | 23 |
| `income_bin` (quantile/K-Means bucket added) | +1 | 24 |
| `high_credit_score` (binarization flag) | +1 | 25 |
| `debt_to_income_ratio` (constructed) | +1 | 26 |
| `avg_monthly_transactions` (constructed) | +1 | **27** |

*(`education_level`, `gender`, `employment_type` are encoded **in place** — no new columns; `spending_ratio` is retained as-is.)*

<br/>

## 💾 Final Dataset

The final processed dataset was exported as:

```
final_cleaned_transformed_dataset.csv
```

<div align="center">

```
┌────────────────────────────────────┐
│           FINAL DATASET             │
├──────────────────────────────────────┤
│ Records          : 500              │
│ Original Columns : 15               │
│ Final Columns    : 27               │
│ Missing Values   : 0                │
│ Duplicates       : 0                │
│ Target           : default_flag     │
│ Status           : ✅ ML READY       │
└──────────────────────────────────────┘
```

</div>

<br/>

## 🗂️ Project Structure

```
customer-credit-risk/
│
├── 📓 Customer_Credit_Risk_Preprocessing.ipynb
│
├── 📊 data/
│   ├── customer_credit_risk_dataset.csv       # 500 × 15 raw
│   ├── customer_metadata.json
│   └── loan_repayment_history.sql
│
├── 🌐 api/
│   └── dummy_economic_api.py
│
├── 📁 output/
│   └── final_cleaned_transformed_dataset.csv  # 500 × 27 final
│
├── 🖼️ assets/
│   ├── workflow.png
│   └── preprocessing_workflow.gif
│
└── 📄 README.md
```

<br/>

## 🖼️ Visualizations

The notebook includes visual comparisons for preprocessing steps such as:

- 📊 Missing value heatmap — before vs after imputation
- 🔤 Encoding comparison — ordinal vs one-hot column expansion
- 📦 Binning distribution — quantile vs K-Means income buckets
- 📏 Scaling comparison — raw vs standardized/normalized features
- 🎯 Target class balance (`default_flag`)
- 🧮 Correlation heatmap of engineered features

> 💡 GitHub renders repository-hosted PNG/SVG/GIF files directly. Replace the placeholders in `assets/` with your actual exported notebook visuals.

<br/>

## 🚀 How to Run

```mermaid
flowchart LR
    A[1️⃣ Clone Repo] --> B[2️⃣ Install Libraries]
    B --> C[3️⃣ Open Jupyter]
    C --> D[4️⃣ Run Notebook Top→Bottom]
    D --> E[5️⃣ Get Final CSV]

    style E fill:#00B894,color:#fff
```

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd customer-credit-risk
```

**2. Install required libraries**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests flask
```

**3. Open Jupyter Notebook**
```bash
jupyter notebook
```
Open `Customer_Credit_Risk_Preprocessing.ipynb`

**4. Run the notebook**
Run cells top to bottom so each preprocessing stage executes in sequence.

**5. Final output**
```
output/final_cleaned_transformed_dataset.csv   → 500 rows × 27 columns
```

<br/>

## 📚 Techniques Covered

<details>
<summary><b>Click to expand full technique list</b></summary>

<br/>

**Data Understanding**
- Pandas profiling · `info()` · `describe()` · Dataset inspection

**Missing Values**
- Mean/Median Imputation · Most Frequent Imputation · Random Sample Imputation · Missing Indicator · KNN Imputation · MICE · Complete Case Analysis

**Outlier Handling**
- Z-Score · IQR · Percentile · Winsorization

**Encoding**
- Ordinal Encoding · Label/Binary Encoding · One-Hot Encoding

**Numerical Feature Engineering**
- Quantile Binning · K-Means Binning · Binarization

**Scaling**
- Standardization · Normalization · Min-Max · MaxAbs · Robust Scaling

**Transformations**
- Log · Reciprocal · Square Root · Box-Cox · Yeo-Johnson

**Feature Construction**
- Debt-to-Income Ratio · Average Monthly Transactions · Spending-to-Income Ratio

**Pipeline-Oriented Processing**
- ColumnTransformer · sklearn Pipeline

</details>

<br/>

## 🎓 Key Learnings

```mermaid
flowchart TD
    A[Raw Data] --> B[Understand]
    B --> C[Clean]
    C --> D[Impute]
    D --> E[Detect Outliers]
    E --> F[Treat Outliers]
    F --> G[Encode]
    G --> H[Bin / Binarize]
    H --> I[Scale]
    I --> J[Transform]
    J --> K[Engineer Features]
    K --> L[Validate]
    L --> M[Final Dataset]

    style A fill:#FF6B6B,color:#fff
    style M fill:#00B894,color:#fff
```

This project demonstrates how raw customer data can be systematically converted into a structured, validated dataset suitable for Machine Learning modeling.

<br/>

## ✅ Final Outcome

The project successfully demonstrates a complete Data Preprocessing and Feature Engineering workflow for a Customer Credit Risk problem.

**Final deliverable: 500 records × 27 columns**

- ✅ Missing values handled (0 remaining)
- ✅ Duplicate rows checked (0 remaining)
- ✅ Outliers treated (Z-Score, IQR, Percentile, Winsorization)
- ✅ Categorical variables encoded (Ordinal, Label, One-Hot)
- ✅ Numerical features processed & scaled
- ✅ Required scaling methods demonstrated
- ✅ Required transformations demonstrated (Log, Box-Cox, Yeo-Johnson)
- ✅ Binning and binarization demonstrated
- ✅ New financial/behavioural features constructed
- ✅ Final CSV generated (`500 × 27`)
- ✅ Target variable preserved (`default_flag`)
- ✅ Dataset fully prepared for Machine Learning

<div align="center">

**DATA UNDERSTANDING → DATA CLEANING → FEATURE ENGINEERING → ML READINESS**

</div>

<br/>

## 👨‍💻 Author

<div align="center">

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=600&size=24&duration=2200&pause=800&color=FD79A8&center=true&vCenter=true&width=650&lines=Hi+%F0%9F%91%8B+I'm+Roshan+Marathe;Data+Preprocessing+%7C+Feature+Engineering;Machine+Learning+Enthusiast;Turning+Raw+Data+Into+ML-Ready+Gold" alt="Typing SVG"/>

<br/>

*Data Preprocessing • Feature Engineering • Machine Learning*

<br/>

<img src="https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
<img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
<img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter"/>
<img src="https://img.shields.io/badge/Instagram-Follow-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram"/>
<img src="https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white"/>

<br/><br/>

<img src="https://github-readme-stats.vercel.app/api?username=Roshan-Marathe&show_icons=true&theme=radical&hide_border=true&count_private=true" height="165" alt="GitHub Stats"/>
<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Roshan-Marathe&layout=compact&theme=radical&hide_border=true" height="165" alt="Top Languages"/>

<br/><br/>

<img src="https://raw.githubusercontent.com/Roshan-Marathe/Roshan-Marathe/output/github-snake.svg" alt="Contribution Snake Animation" width="100%"/>

<br/>

<img src="https://komarev.com/ghpvc/?username=Roshan-Marathe&style=for-the-badge&color=6C5CE7" alt="Profile Views"/>

</div>

> ⚙️ **Setup notes:** Swap every `Roshan-Marathe` above with your real GitHub username, add your actual LinkedIn/Twitter/Instagram/Email links, and set up the [contribution-snake GitHub Action](https://github.com/Platane/snk) on your profile repo so the animated snake renders. GitHub-stats and view-counter images are third-party services — double-check they load correctly after you push.

<br/>

## ⭐ Support This Project

If this project helped you understand the complete preprocessing workflow, consider **starring the repository** ⭐

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=footer" width="100%"/>

</div>
