import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder

# Define file paths
data_path = r"C:\\Users\\cavus\\Desktop\\Forecasting Paper"
results_path = r"C:\\Users\\cavus\\Desktop\\Forecasting Paper\\Results"

# Load datasets
file_west_midlands = f"{data_path}\\Survey_West_Midlands.xlsx"
file_newcastle = f"{data_path}\\Survey_Newcastle.xlsx"

# Read Excel files
west_midlands_data = pd.ExcelFile(file_west_midlands)
newcastle_data = pd.ExcelFile(file_newcastle)

# Parse relevant sheets
west_midlands_df = west_midlands_data.parse("Form Responses 1")
newcastle_df = newcastle_data.parse("Form Responses 1")

# Add region labels to distinguish datasets
west_midlands_df["Region"] = "West Midlands"
newcastle_df["Region"] = "North East"

# Function to format long text labels for better visualization
def format_labels(value):
    return value.replace(" ", "\n")

# Apply label formatting to categorical columns
for df in [west_midlands_df, newcastle_df]:
    df['Education'] = df['Education'].apply(format_labels)
    df['Occupation'] = df['Occupation'].apply(format_labels)
    df['Annual Household Income'] = df['Annual Household Income'].apply(lambda x: x.replace(",", ""))
    df['Ethnic group'] = df['Ethnic group'].replace({
        "Black / African / Caribbean": "Black\nAfrican\nCaribbean"
    })

# Encode categorical variables for violin plots
label_encoders = {}
categorical_columns = ['Education', 'Occupation', 'Annual Household Income', 'Ethnic group']
for col in categorical_columns:
    le = LabelEncoder()
    for df in [west_midlands_df, newcastle_df]:
        df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate DataFrames for analysis
df_wm = west_midlands_df
df_ne = newcastle_df

# Function for plotting violin plots with both regions in one box
def plot_combined_violin(category, title, ylabel):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(x="Region", y=category, data=pd.concat([df_wm, df_ne]), palette="coolwarm", ax=ax)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel("Region", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    
    # Set y-axis labels to actual values instead of numbers for categorical data
    if category in label_encoders:
        labels = label_encoders[category].classes_
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{results_path}\\Violin_{category}_Comparison.png", dpi=600)
    plt.show()

# Generate violin plots for each demographic category
plot_combined_violin("Gender", "Gender Distribution - West Midlands vs North East", "Gender")
plot_combined_violin("Age", "Age Distribution - West Midlands vs North East", "Age")
plot_combined_violin("Education_Encoded", "Education Level Distribution - West Midlands vs North East", "Education Level")
plot_combined_violin("Occupation_Encoded", "Employment Status Distribution - West Midlands vs North East", "Employment Status")
plot_combined_violin("Annual Household Income_Encoded", "Annual Household Income Distribution - West Midlands vs North East", "Salary Range (£)")
plot_combined_violin("Ethnic group_Encoded", "Ethnic Group Distribution - West Midlands vs North East", "Ethnic Group")