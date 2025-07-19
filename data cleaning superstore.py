import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.backends.backend_pdf import PdfPages

# --- Custom Dark Theme for Plots ---
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "#666666",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.labelcolor": "white",
    "grid.color": "#333333",
    "grid.linestyle": "--",
    "figure.titlesize": 12,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "font.size": 8,
    "lines.linewidth": 2,
    "patch.edgecolor": "white",
})

# --- Function for truncating labels ---
def truncate_label(label, max_length=40):
    if len(label) > max_length:
        return label[:max_length - 3] + '...'
    return label

# --- 1. Load the Dataset ---
# Assuming 'superstore single.xls' contains the 'Orders' sheet
try:
    df = pd.read_excel('superstore single.xls', sheet_name='Orders')
    print("Dataset 'Orders' loaded successfully from 'superstore single.xls'!")
except FileNotFoundError:
    print("Error: 'superstore single.xls' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# --- Initialize PDF for saving plots ---
print("All plots will be saved to a PDF file")
pdf_filename = "Superstore_Analysis_Plots.pdf"
pdf_pages = PdfPages(pdf_filename)
print(f"All plots will be saved to '{pdf_filename}'")


# --- 2. Initial Inspection & Overview (Retained for completeness) ---
print("--- 2. Initial Inspection & Overview ---")
print("2.1 Head of the DataFrame:")
print(df.head())
print("2.2 DataFrame Info (Data Types & Non-Null Counts):")
df.info()
print("2.3 Descriptive Statistics for Numerical Columns:")
print(df.describe())
print("2.4 Shape of the DataFrame (Rows, Columns):")
print(df.shape)

# --- 3. Missing Values Inspection (Retained) ---
print("--- 3. Missing Values Inspection ---")
print("3.1 Number of Missing Values per Column:")
print(df.isnull().sum())
# No heatmap generated here to keep output concise for analysis focus.

# --- 4. Duplicate Values Inspection (Retained) ---
print("--- 4. Duplicate Values Inspection ---")
print("4.1 Number of Duplicate Rows:")
print(df.duplicated().sum())
if df.duplicated().sum() > 0:
    print("4.2 Displaying Duplicate Rows (first 5 if many):")
    print(df[df.duplicated()].head())
else:
    print("No duplicate rows found.")
print("4.3 Duplicates based on 'Order ID' column:")
order_id_duplicates = df.duplicated(subset=['Order ID']).sum()
print(order_id_duplicates)
if order_id_duplicates > 0:
    print(df[df.duplicated(subset=['Order ID'], keep=False)].sort_values('Order ID').head())
else:
    print("No duplicate 'Order ID' found.")

# --- 5. Data Type Consistency and Inspection (Retained) ---
print("--- 5. Data Type Consistency and Inspection ---")
print("5.1 Current Data Types:")
print(df.dtypes)
print("5.2 High Cardinality Columns (more than 75 unique values in object type):")
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() > 75:
        print(f"- '{col}' ({df[col].nunique()} unique values)")

# --- 6. Outlier Inspection (for Numerical Columns) (Retained) ---
print("--- 6. Outlier Inspection (for Numerical Columns) ---")
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude 'Row ID' and 'Postal Code' as they are identifiers/codes, not metrics for outlier capping
cols_for_outlier_check = [col for col in numerical_cols if col not in ['Row ID', 'Postal Code']]

for col in cols_for_outlier_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    if outliers_count > 0:
        print(f"- Column '{col}': {outliers_count} potential outliers (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
    else:
        print(f"- Column '{col}': No obvious outliers detected by IQR method.")

# No box plots shown here to keep output concise for analysis focus.

# --- 7. Whitespace and Case Consistency (Retained) ---
print("--- 7. Whitespace and Case Consistency ---")
for col in df.select_dtypes(include='object').columns:
    if df[col].apply(lambda x: isinstance(x, str) and (x.strip() != x)).any():
        print(f"- Column '{col}': Contains leading/trailing spaces.")
    original_unique_count = df[col].nunique()
    cleaned_unique_count = df[col].astype(str).str.lower().nunique()
    if original_unique_count != cleaned_unique_count:
        print(f"- Column '{col}': Contains case inconsistencies ({original_unique_count} original vs {cleaned_unique_count} lowercased unique values).")

# # --- 8. Date Column Inspection and Conversion (Enhanced) ---
# print("--- 8. Date Column Inspection and Conversion ---")
# date_cols = ['Order Date', 'Ship Date'] # Explicitly define date columns for Superstore
# for col in date_cols:
#     if col in df.columns:
#         # Convert to datetime, coercing errors to NaT (Not a Time)
#         df[col] = pd.to_datetime(df[col], errors='coerce', unit='D', origin='1899-12-30') # Excel date origin
#         if df[col].isnull().any():
#             print(f"  Warning: Column '{col}' has unparseable date values after conversion (NaT introduced).")
#         print(f"  Min Date for '{col}': {df[col].min()}, Max Date for '{col}': {df[col].max()}")
#     else:
#         print(f"  Column '{col}' not found in the dataset.")
#
# # Drop rows where essential date columns are NaT if necessary for analysis
# df.dropna(subset=date_cols, inplace=True)
# print(f"Dropped rows with missing values in {date_cols}. New shape: {df.shape}")

# --- 9. Sales and Profit Trends Over Time ---
print("--- 9. Sales and Profit Trends Over Time ---")

# Aggregate sales and profit by month and year
df['Order_YearMonth'] = df['Order Date'].dt.to_period('M')
monthly_trends = df.groupby('Order_YearMonth')[['Sales', 'Profit']].sum().reset_index()
monthly_trends['Order_YearMonth'] = monthly_trends['Order_YearMonth'].astype(str) # Convert Period to string for plotting

print("Monthly Sales and Profit Trends (first 5 rows):")
print(monthly_trends.head())

# Plotting Monthly Sales Trend
fig_monthly_sales = plt.figure(figsize=(14, 7))
sns.lineplot(x='Order_YearMonth', y='Sales', data=monthly_trends, marker='o', color='skyblue')
plt.title('Monthly Sales Trend Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig_monthly_sales)
plt.close(fig_monthly_sales)

# Plotting Monthly Profit Trend
fig_monthly_profit = plt.figure(figsize=(14, 7))
sns.lineplot(x='Order_YearMonth', y='Profit', data=monthly_trends, marker='o', color='lightcoral')
plt.title('Monthly Profit Trend Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Total Profit')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig_monthly_profit)
plt.close(fig_monthly_profit)

# --- 10. Sales and Profit by Product Category ---
print("--- 10. Sales and Profit by Product Category ---")

category_performance = df.groupby('Category')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False)
print("Sales and Profit by Product Category:")
print(category_performance)

# Plotting Sales by Category
fig_category_sales = plt.figure(figsize=(10, 6))
sns.barplot(x=category_performance.index, y='Sales', data=category_performance, palette='viridis')
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
pdf_pages.savefig(fig_category_sales)
plt.close(fig_category_sales)

# Plotting Profit by Category
fig_category_profit = plt.figure(figsize=(10, 6))
sns.barplot(x=category_performance.index, y='Profit', data=category_performance, palette='magma')
plt.title('Total Profit by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Profit')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
pdf_pages.savefig(fig_category_profit)
plt.close(fig_category_profit)

# --- 11. Sales and Profit by Sub-Category ---
print("--- 11. Sales and Profit by Sub-Category ---")

subcategory_performance = df.groupby('Sub-Category')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False)
print("Sales and Profit by Sub-Category (Top 10):")
print(subcategory_performance.head(10))

# Plotting Sales by Sub-Category (Top N for readability)
top_n_subcategories = 15
fig_subcategory_sales = plt.figure(figsize=(14, 8))
sns.barplot(x=subcategory_performance.head(top_n_subcategories).index,
            y='Sales',
            data=subcategory_performance.head(top_n_subcategories),
            palette='cividis')
plt.title(f'Top {top_n_subcategories} Sub-Categories by Sales')
plt.xlabel('Sub-Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
pdf_pages.savefig(fig_subcategory_sales)
plt.close(fig_subcategory_sales)

# Plotting Profit by Sub-Category (Top N for readability, also showing negative profit)
fig_subcategory_profit = plt.figure(figsize=(14, 8))
sns.barplot(x=subcategory_performance.head(top_n_subcategories).index,
            y='Profit',
            data=subcategory_performance.head(top_n_subcategories),
            palette='plasma')
plt.title(f'Top {top_n_subcategories} Sub-Categories by Profit')
plt.xlabel('Sub-Category')
plt.ylabel('Total Profit')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
pdf_pages.savefig(fig_subcategory_profit)
plt.close(fig_subcategory_profit)

# Identify sub-categories with negative profit
negative_profit_subcategories = subcategory_performance[subcategory_performance['Profit'] < 0]
if not negative_profit_subcategories.empty:
    print("Sub-Categories with Negative Profit:")
    print(negative_profit_subcategories.sort_values(by='Profit', ascending=True))
    fig_negative_profit = plt.figure(figsize=(12, 7))
    sns.barplot(x=negative_profit_subcategories.index,
                y='Profit',
                data=negative_profit_subcategories,
                palette='Reds_d')
    plt.title('Sub-Categories with Negative Profit')
    plt.xlabel('Sub-Category')
    plt.ylabel('Total Profit')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    pdf_pages.savefig(fig_negative_profit)
    plt.close(fig_negative_profit)
else:
    print("No sub-categories with negative profit found.")

# --- 12. Impact of Discount on Profit ---
print("--- 12. Impact of Discount on Profit ---")

# Binning discount rates for better analysis
bins = [-0.01, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 0.0 included for no discount
labels = ['No Discount (0%)', '0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
df['Discount_Group'] = pd.cut(df['Discount'], bins=bins, labels=labels, right=True)

discount_profit = df.groupby('Discount_Group')[['Sales', 'Profit']].sum().reset_index()
print("Sales and Profit by Discount Group:")
print(discount_profit)

# Plotting Profit by Discount Group
fig_discount_profit = plt.figure(figsize=(12, 7))
sns.barplot(x='Discount_Group', y='Profit', data=discount_profit, palette='coolwarm')
plt.title('Total Profit by Discount Group')
plt.xlabel('Discount Group')
plt.ylabel('Total Profit')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
pdf_pages.savefig(fig_discount_profit)
plt.close(fig_discount_profit)

# --- 13. Sales and Profit by Region ---
print("--- 13. Sales and Profit by Region ---")

region_performance = df.groupby('Region')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False)
print("Sales and Profit by Region:")
print(region_performance)

# Plotting Sales by Region
fig_region_sales = plt.figure(figsize=(10, 6))
sns.barplot(x=region_performance.index, y='Sales', data=region_performance, palette='viridis')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.tight_layout()
pdf_pages.savefig(fig_region_sales)
plt.close(fig_region_sales)

# Plotting Profit by Region
fig_region_profit = plt.figure(figsize=(10, 6))
sns.barplot(x=region_performance.index, y='Profit', data=region_performance, palette='magma')
plt.title('Total Profit by Region')
plt.xlabel('Region')
plt.ylabel('Total Profit')
plt.tight_layout()
pdf_pages.savefig(fig_region_profit) # Save the figure to PDF
plt.close(fig_region_profit) # Close the figure to free memory

# --- 14. Additional Key Performance Indicators (KPIs) ---
print("--- 14. Additional Key Performance Indicators (KPIs) ---")

# Overall Profit Margin
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
overall_profit_margin = (total_profit / total_sales) * 100 if total_sales != 0 else 0
print(f"Overall Profit Margin: {overall_profit_margin:.2f}%")

# Average Order Value
# An order can have multiple rows (products), so group by 'Order ID' first
order_values = df.groupby('Order ID')['Sales'].sum()
average_order_value = order_values.mean()
print(f"Average Order Value: ${average_order_value:.2f}")

# Top 10 Customers by Sales and Profit
customer_performance = df.groupby('Customer Name')[['Sales', 'Profit']].sum()

top_10_customers_sales = customer_performance.sort_values(by='Sales', ascending=False).head(10)
print("Top 10 Customers by Sales:")
print(top_10_customers_sales)

top_10_customers_profit = customer_performance.sort_values(by='Profit', ascending=False).head(10)
print("Top 10 Customers by Profit:")
print(top_10_customers_profit)

# Top 10 Products by Sales and Profit
product_performance = df.groupby('Product Name')[['Sales', 'Profit']].sum()

top_10_products_sales = product_performance.sort_values(by='Sales', ascending=False).head(10)
print("Top 10 Products by Sales:")
print(top_10_products_sales)

top_10_products_profit = product_performance.sort_values(by='Profit', ascending=False).head(10)
print("Top 10 Products by Profit:")
print(top_10_products_profit)

# Sales and Profit per Customer Segment
segment_performance = df.groupby('Segment')[['Sales', 'Profit']].sum()
print("Sales and Profit by Customer Segment:")
print(segment_performance)

# Average Sales and Profit per Quantity
# Calculate these per line item, then average
df['Sales_Per_Quantity'] = df['Sales'] / df['Quantity']
df['Profit_Per_Quantity'] = df['Profit'] / df['Quantity']

avg_sales_per_quantity = df['Sales_Per_Quantity'].mean()
avg_profit_per_quantity = df['Profit_Per_Quantity'].mean()

print(f"Average Sales per Unit Quantity: ${avg_sales_per_quantity:.2f}")
print(f"Average Profit per Unit Quantity: ${avg_profit_per_quantity:.2f}")

# --- Close the PDF file ---
pdf_pages.close()
print(f"All plots have been saved to '{pdf_filename}'")
print("--- Enhanced Superstore Data Analysis Complete ---")
