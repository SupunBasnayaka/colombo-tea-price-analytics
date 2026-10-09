import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set global visual style for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_dataset():
    """Load dataset from Hugging Face CSV or datasets library."""
    print("==================================================")
    print("STEP 1: LOADING DATASET")
    print("==================================================")
    
    url = "https://huggingface.co/datasets/colombo-tea-auction-prices/cta-price-prediction/raw/main/train.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        from datasets import load_dataset
        dataset = load_dataset("colombo-tea-auction-prices/cta-price-prediction")
        df = pd.DataFrame(dataset['train'])

    print(f"Dataset successfully loaded. Shape: {df.shape}\n")
    print("First 5 rows:")
    print(df.head())
    print("\n")
    return df


def eda_and_preprocessing(df):
    """Perform EDA, check data quality, and handle missing values."""
    print("==================================================")
    print("STEP 2: EXPLORATORY DATA ANALYSIS & DATA CLEANING")
    print("==================================================")

    # Check for missing values
    missing_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df)) * 100
    })
    print("Missing Values Summary:")
    print(missing_info)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Imputation Strategy
    print("\nImputing missing values...")
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values after imputation:", df.isnull().sum().sum())

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Price Distribution
    sns.histplot(df['price_mid_lkr'], kde=True, ax=axes[0, 0], color='teal')
    axes[0, 0].set_title("1. Distribution of Mid Prices (LKR)")

    # 2. Price across Elevations
    if 'elevation' in df.columns:
        sns.boxplot(data=df, x='elevation', y='price_mid_lkr', ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title("2. Tea Price Distribution across Elevations")

    # 3. USD Exchange Rate vs Price
    if 'fx_usd' in df.columns:
        sns.scatterplot(data=df, x='fx_usd', y='price_mid_lkr', alpha=0.5, ax=axes[1, 0], color='coral')
        axes[1, 0].set_title("3. Exchange Rate (USD/LKR) vs Price")

    # 4. Correlation Heatmap
    weather_cols = [c for c in num_cols if any(k in c for k in ['precipitation', 'temperature', 'sunshine', 'price_mid_lkr'])]
    if len(weather_cols) > 1:
        sns.heatmap(df[weather_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
        axes[1, 1].set_title("4. Correlation Heatmap: Weather & Price")

    plt.tight_layout()
    print("\nDisplaying EDA Plots... (Close the plot window to continue)")
    plt.show()

    return df, num_cols, cat_cols


def run_hypothesis_testing(df):
    """Execute formal statistical tests (ANOVA & Pearson Correlation)."""
    print("\n==================================================")
    print("STEP 3: STATISTICAL INFERENCE & HYPOTHESIS TESTING")
    print("==================================================")

    # Test 1: One-way ANOVA - Price differences across Elevations
    if 'elevation' in df.columns:
        groups = [group['price_mid_lkr'].values for _, group in df.groupby('elevation')]
        f_stat, p_val_anova = stats.f_oneway(*groups)
        print("1. One-Way ANOVA Test (Price across Elevation Groups):")
        print(f"   F-Statistic = {f_stat:.4f}, p-value = {p_val_anova:.4e}")
        if p_val_anova < 0.05:
            print("   Conclusion: Reject H0. Elevation has a statistically significant effect on tea prices.\n")
        else:
            print("   Conclusion: Fail to reject H0. No significant price difference across elevations.\n")

    # Test 2: Correlation Significance (USD FX vs Price)
    if 'fx_usd' in df.columns:
        corr, p_val_corr = stats.pearsonr(df['fx_usd'], df['price_mid_lkr'])
        print("2. Pearson Correlation Test (USD Exchange Rate vs Price):")
        print(f"   Correlation Coefficient = {corr:.4f}, p-value = {p_val_corr:.4e}")

        # 95% Confidence Interval calculation
        z = np.arctanh(corr)
        sigma = 1 / np.sqrt(len(df) - 3)
        c_int = np.tanh([z - 1.96 * sigma, z + 1.96 * sigma])
        print(f"   95% Confidence Interval for Correlation: [{c_int[0]:.4f}, {c_int[1]:.4f}]\n")


def build_and_evaluate_models(df):
    """Train regression models and evaluate with standard metrics."""
    print("==================================================")
    print("STEP 4: MODELING, EVALUATION & INNOVATIVE ANALYSIS")
    print("==================================================")

    target = 'price_mid_lkr'
    ignore_cols = ['sale_id', 'sale_date_raw', target]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df[target]

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'RMSE (LKR)': round(rmse, 2),
            'MAE (LKR)': round(mae, 2),
            'R² Score': round(r2, 4)
        })

    results_df = pd.DataFrame(results)
    print("Model Evaluation Summary:")
    print(results_df)

    # Innovative Approach: Residual Diagnostic Analysis
    best_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    best_pipe.fit(X_train, y_train)

    y_pred_best = best_pipe.predict(X_test)
    residuals = y_test - y_pred_best

    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred_best, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price (LKR)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Innovative Diagnostic: Residual Analysis for Model Calibration")
    
    print("\nDisplaying Residual Diagnostics Plot... (Close plot to finish script)")
    plt.show()


if __name__ == "__main__":
    data = load_dataset()
    cleaned_data, num_cols, cat_cols = eda_and_preprocessing(data)
    run_hypothesis_testing(cleaned_data)
    build_and_evaluate_models(cleaned_data)
    print("\nProject execution complete!")
