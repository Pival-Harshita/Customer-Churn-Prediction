import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# 1. DATASET LOADING
# ---------------------------------

def acquire_data(filepath):

    print("\nDATASET LOADING\n")

    df = pd.read_csv(filepath)

    print("Dataset loaded successfully")

    print("\nShape of dataset:", df.shape)

    print("\nFirst 5 rows:\n")
    print(df.head())

    print("\nLast 5 rows:\n")
    print(df.tail())

    print("\nColumn names in dataset:")
    print(df.columns)

    return df


# ---------------------------------
# 2. STRUCTURAL UNDERSTANDING
# ---------------------------------

def structural_understanding(df):

    print("\nSTRUCTURAL UNDERSTANDING\n")

    print("Data types of each column:")
    print(df.dtypes)

    print("\nDataset information:")
    print(df.info())

    print("\nUnique value count per column:")
    print(df.nunique())


def identify_identifier_columns(df):

    print("\nChecking for potential identifier columns...\n")

    identifier_columns = []

    for col in df.columns:

        unique_values = df[col].nunique()

        if unique_values == len(df):
            identifier_columns.append(col)

    if identifier_columns:

        print("Potential Identifier Columns:")

        for col in identifier_columns:
            print(f"- {col}")

        print("\nThese columns have unique values for each row.")
        print("They act as identifiers and usually do not contribute to prediction.")

    else:

        print("No identifier columns detected.")

    return identifier_columns


# ---------------------------------
# 3. STATISTICAL UNDERSTANDING
# ---------------------------------

def statistical_understanding(df):

    print("\nSTATISTICAL UNDERSTANDING\n")

    print("Statistical Summary:\n")

    print(df.describe())


def feature_distribution_skewness(df):

    print("\nFeature Skewness Values:\n")

    numeric_columns = df.select_dtypes(include=['int64','float64']).columns

    skewness_values = df[numeric_columns].skew()

    print(skewness_values)

    for col in numeric_columns:

        plt.figure()

        sns.histplot(df[col], kde=True)

        plt.title(f"Distribution of {col}")

        plt.xlabel(col)

        plt.ylabel("Frequency")

        plt.show()


# ---------------------------------
# 4. DATA QUALITY AWARENESS
# ---------------------------------

def data_quality_awareness(df):

    print("\nDATA QUALITY AWARENESS\n")

    print("Missing Values:\n")

    print(df.isnull().sum())


def check_duplicate_records(df):

    print("\nChecking Duplicate Records...\n")

    duplicate_count = df.duplicated().sum()

    print("Number of duplicate records:", duplicate_count)

    if duplicate_count > 0:

        print("\nDuplicate rows found:")

        print(df[df.duplicated()].head())

    else:

        print("No duplicate records found in the dataset.")


# ---------------------------------
# 5. TARGET ANALYSIS
# ---------------------------------

def target_analysis(df, target_column):

    print("\nTARGET VARIABLE ANALYSIS\n")

    target_counts = df[target_column].value_counts()

    print("Target Class Counts:\n")

    print(target_counts)

    target_percentage = df[target_column].value_counts(normalize=True) * 100

    print("\nTarget Class Percentage:\n")

    print(target_percentage)

    ratio = target_counts.iloc[0] / target_counts.iloc[1]

    print("\nClass Ratio (Majority : Minority):")

    print(f"{target_counts.index[0]} : {target_counts.index[1]} = {ratio:.2f} : 1")

    plt.figure()

    sns.countplot(x=target_column, data=df)

    plt.title("Target Variable Distribution")

    plt.xlabel(target_column)

    plt.ylabel("Count")

    plt.show()


def visualize_class_balance(df, target_column):

    print("\nVisualizing Class Balance...\n")

    plt.figure(figsize=(6,4))

    sns.countplot(x=target_column, data=df)

    plt.title("Class Distribution")

    plt.xlabel("Target Class")

    plt.ylabel("Count")

    plt.show()

    class_counts = df[target_column].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(class_counts,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90)

    plt.title("Class Distribution Percentage")

    plt.show()


# ---------------------------------
# MAIN FUNCTION
# ---------------------------------

def main():

    filepath = "D:/Churn_Prediction/dataset/github_dataset.csv"

    target_column = "Churn"

    df = acquire_data(filepath)

    structural_understanding(df)

    identify_identifier_columns(df)

    statistical_understanding(df)

    feature_distribution_skewness(df)

    data_quality_awareness(df)

    check_duplicate_records(df)

    target_analysis(df, target_column)

    visualize_class_balance(df, target_column)


if __name__ == "__main__":
    main()