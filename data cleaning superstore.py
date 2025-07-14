import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Dataset ---

try:
    df = pd.read_excel('superstore single.xls')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'your_dataset.csv' not found. Please check the file path.")
    exit()

# --- 2. Initial Inspection & Overview ---
print("--- 2. Initial Inspection & Overview ---")

# Display the first few rows of the DataFrame
print("2.1 Head of the DataFrame:")
print(df.head())

# Display the last few rows
print("2.2 Tail of the DataFrame:")
print(df.tail())

# Get a concise summary of the DataFrame, including data types and non-null values
print("2.3 DataFrame Info (Data Types & Non-Null Counts):")
df.info()

# Get descriptive statistics for numerical columns
print("2.4 Descriptive Statistics for Numerical Columns:")
print(df.describe())

# Get descriptive statistics for all columns, including categorical
print("2.5 Descriptive Statistics for All Columns (including categorical):")
print(df.describe(include='all'))

# Get the shape of the DataFrame (number of rows, number of columns)
print("2.6 Shape of the DataFrame (Rows, Columns):")
print(df.shape)

# Get the column names
print("2.7 Column Names:")
print(df.columns.tolist())

# --- 3. Missing Values Inspection ---
print("--- 3. Missing Values Inspection ---")

# Check for missing values in each column
print("3.1 Number of Missing Values per Column:")
print(df.isnull().sum())

# Check for missing values as a percentage
print("3.2 Percentage of Missing Values per Column:")
print((df.isnull().sum() / len(df)) * 100)

# Visualize missing values (requires matplotlib and seaborn)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# --- 4. Duplicate Values Inspection ---
print("--- 4. Duplicate Values Inspection ---")

# Check for duplicate rows
print("4.1 Number of Duplicate Rows:")
print(df.duplicated().sum())

# Display the duplicate rows (if any)
if df.duplicated().sum() > 0:
    print("4.2 Displaying Duplicate Rows (first 5 if many):")
    print(df[df.duplicated()].head())
else:
    print("No duplicate rows found.")

# Check for duplicates based on specific unique columns
print("4.3 Duplicates based on 'OrderID' column:")
print(df.duplicated(subset=['Order ID']).sum())
if df.duplicated(subset=['Order ID']).sum() > 0:
    print(df[df.duplicated(subset=['Order ID'], keep=False)].sort_values('Order ID').head())
else:
    print("No duplicate 'Order ID' found.")


# --- 5. Data Type Consistency and Inspection ---
print("--- 5. Data Type Consistency and Inspection ---")

# Display current data types
print("5.1 Current Data Types:")
print(df.dtypes)

# Check numerical columns that might be objects
print("5.2 Potential Numerical Columns Stored as Objects:")
for col in df.select_dtypes(include='object').columns:
    try:
        # Try converting to numeric
        pd.to_numeric(df[col], errors='coerce')
        if df[col].dtype == 'object':
            print(f"- '{col}' (contains non-numeric values or mixed types)")
    except Exception:
        pass # Not convertible to numeric

# Inspect unique values for object/categorical columns to spot inconsistencies
print("5.3 Unique Values for Categorical/Object Columns (Top 10):")
for col in df.select_dtypes(include='object').columns:
    unique_vals = df[col].unique()
    print(f"Column '{col}' ({len(unique_vals)} unique values):")
    if len(unique_vals) > 10:
        print(f"  {unique_vals[:10]} ... (and {len(unique_vals) - 10} more)")
    else:
        print(f"  {unique_vals}")

# Identify columns with too many unique values (high cardinality)
print("5.4 High Cardinality Columns (more than 50 unique values in object type):")
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() > 75:
        print(f"- '{col}' ({df[col].nunique()} unique values)")


# --- 6. Outlier Inspection (for Numerical Columns) ---
print("--- 6. Outlier Inspection (for Numerical Columns) ---")

print("6.1 Interquartile Range (IQR) and Potential Outliers (for numerical columns):")
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    if outliers_count > 0:
        print(f"- Column '{col}': {outliers_count} potential outliers (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
        print(df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].head())
    else:
        print(f"- Column '{col}': No obvious outliers detected by IQR method.")

# Box plots for visual outlier detection
print("6.2 Generating Box Plots for Numerical Columns (visual outlier detection):")
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()

# --- 7. Whitespace and Case Consistency (for Object/String Columns) ---
print("--- 7. Whitespace and Case Consistency ---")

print("Checking for leading/trailing spaces and inconsistent casing in object columns...")
for col in df.select_dtypes(include='object').columns:
    if df[col].apply(lambda x: isinstance(x, str) and (x.strip() != x)).any():
        print(f"- Column '{col}': Contains leading/trailing spaces.")

    # Check for case inconsistencies (e.g., 'USA' vs 'Usa')
    original_unique_count = df[col].nunique()
    cleaned_unique_count = df[col].astype(str).str.lower().nunique()
    if original_unique_count != cleaned_unique_count:
        print(f"- Column '{col}': Contains case inconsistencies ({original_unique_count} original vs {cleaned_unique_count} lowercased unique values).")


# --- 8. Date Column Inspection ---
print("--- 8. Date Column Inspection ---")

# Identify potential date columns (columns with 'date' in their name or recognized by pandas)
date_cols = [col for col in df.columns if 'date' in col.lower() or 'order_dt' in col.lower()]
if not date_cols:
    # Try to infer date columns based on data type after initial info()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)

if date_cols:
    print("Potential Date Columns identified:")
    for col in date_cols:
        print(f"- {col}")
        # Try to convert to datetime to catch parsing errors
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].isnull().any():
                print(f"  Warning: Column '{col}' has unparseable date values after conversion (NaNs introduced).")
            print(f"  Min Date: {df[col].min()}, Max Date: {df[col].max()}")
        except Exception as e:
            print(f"  Error converting '{col}' to datetime: {e}")
else:
    print("No obvious date columns found.")

print("-- Data Inspection Complete ---")