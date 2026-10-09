# Colombo Tea Auction Price Prediction & Statistical Analysis

This document provides the setup, configuration, and step-by-step execution instructions for running the Colombo Tea Auction Price Prediction project.

---

## Table of Contents
1. [Project Overview & Structure](#1-project-overview--structure)
2. [Prerequisites](#2-prerequisites)
3. [Configuration & Dependencies (`requirements.txt`)](#3-configuration--dependencies-requirementstxt)
4. [Step-by-Step Execution Guide](#4-step-by-step-execution-guide)
5. [Troubleshooting & FAQs](#5-troubleshooting--faqs)

---

## 1. Project Overview & Structure

### Problem Summary
Predicting Colombo Tea Auction mid-prices (`price_mid_lkr`) based on macro-economic drivers (`USD exchange rate`), geographical attributes (`elevation, grade`,` region`), and lagged meteorological factors (`precipitation`, `sunshine`, `temperature`).

### Expected Project Files
Your project directory should be structured as follows:

```text
tea_price_project/
├── main.py    # Main Python application script
├── requirements.txt         # Package configuration file
└── PROJECT.md               # Configuration and setup guide
```

---

## 2. Prerequisites

Before running the project, ensure you have installed:
* **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
* **VS Code**: [Download Visual Studio Code](https://code.visualstudio.com/)
* **Python Extension for VS Code**: Installed via the VS Code Extensions tab (`Ctrl + Shift + X` / `Cmd + Shift + X`).

---

## 3. Configuration & Dependencies (`requirements.txt`)

Create a file named `requirements.txt` in your project folder with the following dependencies:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
requests
datasets
huggingface-hub
pyarrow
fastparquet
jupyter
notebook
```

---

## 4. Step-by-Step Execution Guide

### Step 1: Open Your Workspace in VS Code
1. Open **VS Code**.
2. Go to **File > Open Folder...** and select your project folder (`tea_price_project`).
3. Confirm that `main.py` and `requirements.txt` are located inside this folder.

### Step 2: Open the Integrated Terminal
Open the VS Code terminal by pressing **`Ctrl + ~`** (or selecting **Terminal > New Terminal** from the top menu bar).

### Step 3: Create & Activate a Virtual Environment
Creating an isolated environment ensures clean package management.

* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

*(When activated, `(venv)` will appear at the start of your terminal prompt line).*

### Step 4: Install Dependencies
Install all required libraries by running:

```bash
pip install -r requirements.txt
```

### Step 5: Execute the Script
Run the script through the terminal:

```bash
python main.py
```

---

## 5. Troubleshooting & FAQs

* **Plot Windows Pause Script Execution:** 
  `matplotlib` displays figures in interactive popup windows. To allow the script to proceed to subsequent steps (such as hypothesis testing or modeling results), click the close **(X)** button on each plot window.
* **`ModuleNotFoundError`:**
  If you encounter an error stating a package is missing, verify that your virtual environment is active (`(venv)` shown in prompt) and re-run `pip install -r requirements.txt`.
* **Dataset Connection Warning:**
  The dataset downloads live from Hugging Face. Ensure your computer is connected to the internet during execution.