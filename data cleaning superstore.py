import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold # Import GridSearchCV and KFold
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

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

# --- 8. Date Column Inspection and Conversion (Enhanced) ---
# Removed as per user request. Date conversion now happens implicitly when reading Excel.
# If explicit conversion is needed for specific columns later, it should be added there.
# For this script, we'll assume pandas handles it well enough for basic operations.

# --- 9. Sales and Profit Trends Over Time ---
print("--- 9. Sales and Profit Trends Over Time ---")

# Ensure 'Order Date' is datetime type for time-series analysis
# Removed unit='D', origin='1899-12-30' as it's causing issues with already parsed dates.
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df.dropna(subset=['Order Date'], inplace=True) # Drop rows where date conversion failed

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

# --- 15. Numerical Analysis: Correlation ---
print("\n--- 15. Numerical Analysis: Correlation ---")

# Select numerical columns for correlation analysis
correlation_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
correlation_matrix = df[correlation_cols].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plotting the correlation heatmap
fig_corr_heatmap = plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
pdf_pages.savefig(fig_corr_heatmap)
plt.close(fig_corr_heatmap)

# --- 16. Numerical Analysis: Multiple Linear Regression ---
print("\n--- 16. Numerical Analysis: Multiple Linear Regression ---")

# Define dependent and independent variables
# Let's try to predict Profit based on Sales, Quantity, and Discount
X = df[['Sales', 'Quantity', 'Discount']]
y = df['Profit']

# Check for perfect multicollinearity (optional but good practice)
# If a column can be perfectly predicted from others, it can cause issues.
# For example, if 'Sales' was always 10 * 'Quantity' * 'Price_per_unit', etc.
# In our case, these are distinct transactional values.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation (Linear Regression):")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Display model coefficients and intercept
print("\nModel Coefficients (Linear Regression):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Using statsmodels for a more detailed statistical summary
# Add a constant (intercept) to the independent variables for statsmodels
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print("\nDetailed Regression Results (Statsmodels OLS):")
print(model_sm.summary())

# --- Add Regression Results to PDF as a Table ---
print("\n--- Adding Linear Regression Results to PDF ---")

# Extract key parts of the summary for the table
results_df = pd.DataFrame({
    'Coefficient': model_sm.params,
    'Std Error': model_sm.bse,
    't-value': model_sm.tvalues,
    'P-value': model_sm.pvalues,
    '[0.025': model_sm.conf_int()[0],
    '0.975]': model_sm.conf_int()[1]
})

# Format for display in table
results_df = results_df.map(lambda x: f"{x:.4f}")

# Add R-squared and Adjusted R-squared
r_squared = f"{model_sm.rsquared:.4f}"
adj_r_squared = f"{model_sm.rsquared_adj:.4f}"
f_statistic = f"{model_sm.fvalue:.2f}"
prob_f_statistic = f"{model_sm.f_pvalue:.4f}"

# Create a figure for the table
fig_regression_table = plt.figure(figsize=(10, 6))
ax = fig_regression_table.add_subplot(111)
ax.axis('off') # Hide axes

# Prepare table data, including a header for the index
header = ['Feature'] + results_df.columns.tolist()
table_data = [header] + results_df.reset_index().values.tolist()

# Create the table
table = ax.table(cellText=table_data,
                 colLabels=None,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2) # Adjust size as needed

# Style the table for dark theme
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('black')
    cell.set_edgecolor('#666666')
    cell.set_text_props(color='white')
    if row == 0: # Header row
        cell.set_text_props(weight='bold')

ax.set_title('Multiple Linear Regression Results', color='white', fontsize=14)

# Add R-squared and Adjusted R-squared below the table as text
ax.text(0.05, 0.15, f'R-squared: {r_squared}', transform=ax.transAxes, fontsize=10, color='white')
ax.text(0.05, 0.10, f'Adj. R-squared: {adj_r_squared}', transform=ax.transAxes, fontsize=10, color='white')
ax.text(0.05, 0.05, f'F-statistic: {f_statistic} (Prob(F-statistic): {prob_f_statistic})', transform=ax.transAxes, fontsize=10, color='white')

plt.tight_layout()
pdf_pages.savefig(fig_regression_table)
plt.close(fig_regression_table)

# --- 17. Numerical Analysis: Random Forest Regression with Hyperparameter Tuning ---
print("\n--- 17. Numerical Analysis: Random Forest Regression with Hyperparameter Tuning ---")

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200], # Number of trees in the forest
    'max_features': [0.6, 0.8, 1.0], # Number of features to consider when looking for the best split
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
    'min_samples_split': [2, 5, 10] # Minimum number of samples required to split an internal node
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Set up KFold for cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42) # Changed to 5 folds

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='r2', verbose=1, n_jobs=-1)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest Parameters found by GridSearchCV: {best_params}")
print(f"Best R-squared score from cross-validation: {best_score:.2f}")

# Train the Random Forest model with the best parameters
tuned_rf_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
tuned_rf_model.fit(X_train, y_train)

# Make predictions on the test set using the tuned model
y_pred_rf_tuned = tuned_rf_model.predict(X_test)

# Evaluate the tuned Random Forest model
mse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned)
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)

print(f"\nModel Evaluation (Tuned Random Forest Regression on Test Set):")
print(f"Mean Squared Error (MSE): {mse_rf_tuned:.2f}")
print(f"R-squared (R2): {r2_rf_tuned:.2f}")

# Feature Importances from the tuned model
feature_importances_tuned = pd.Series(tuned_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (Tuned Random Forest):")
print(feature_importances_tuned)

# Plotting Feature Importances for Tuned Random Forest
fig_feature_importance_tuned = plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_tuned.index, y=feature_importances_tuned.values, palette='viridis')
plt.title('Feature Importances (Tuned Random Forest Regression)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
pdf_pages.savefig(fig_feature_importance_tuned)
plt.close(fig_feature_importance_tuned)

# --- Close the PDF file ---
pdf_pages.close()
print(f"All plots and regression results have been saved to '{pdf_filename}'")
print("--- Enhanced Superstore Data Analysis Complete ---")
