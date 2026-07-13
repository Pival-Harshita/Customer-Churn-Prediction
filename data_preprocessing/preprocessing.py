import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------------------------------

def plot_churn_distribution(df):
    plt.figure(figsize=(5,4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Class Distribution")
    plt.show()


def plot_churn_percentage(df):
    plt.figure(figsize=(5,5))
    df["Churn"].value_counts().plot.pie(
        autopct="%1.1f%%",
        startangle=90,
        colors=["skyblue","salmon"]
    )
    plt.title("Churn Percentage")
    plt.ylabel("")
    plt.show()


def plot_boxplots(df):
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    for col in cols:
        if col in df.columns:
            plt.figure(figsize=(5,3))
            sns.boxplot(y=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()


def plot_correlation(df):
    print("\nPlotting Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Numeric Only)")
    plt.show()


def plot_smote_balance(y_before, y_after):
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    sns.countplot(x=y_before, ax=ax[0])
    ax[0].set_title("Before SMOTE")

    sns.countplot(x=y_after, ax=ax[1])
    ax[1].set_title("After SMOTE")

    plt.show()


# ---------------------------------------------------
# 1 REMOVE IDENTIFIER COLUMNS
# ---------------------------------------------------

def remove_identifier_columns(df):

    print("\nRemoving Identifier Columns")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("Removed: customerID")

    return df


# ---------------------------------------------------
# 2 FIX DATA TYPES
# ---------------------------------------------------

def fix_datatypes(df):

    print("\nFixing Data Types")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


# ---------------------------------------------------
# 3 FEATURE ENGINEERING
# ---------------------------------------------------

def feature_engineering(df):

    print("\nFeature Engineering")

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    if "tenure" in df.columns:
        df["tenure_segment"] = pd.cut(
            df["tenure"],
            bins=[0,12,24,48,72],
            labels=["new","short_term","mid_term","long_term"]
        )

    service_cols = [
        "PhoneService","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies"
    ]

    service_cols = [c for c in service_cols if c in df.columns]

    if service_cols:
        df["service_dependency_index"] = df[service_cols].apply(
            lambda x: sum(x.astype(str).str.contains("Yes")), axis=1
        )

    if "Contract" in df.columns:

        risk_map = {
            "Month-to-month":3,
            "One year":2,
            "Two year":1
        }

        df["contract_risk"] = df["Contract"].map(risk_map)

    return df


# ---------------------------------------------------
# 4 HANDLE MISSING VALUES
# ---------------------------------------------------

def handle_missing_values(df):

    print("\nHandling Missing Values")

    print("\nMissing values before\n")
    print(df.isnull().sum())

    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    cat_cols = df.select_dtypes(include=["object","category"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values after\n")
    print(df.isnull().sum())

    return df


# ---------------------------------------------------
# 5 OUTLIER DETECTION
# ---------------------------------------------------

def detect_outliers(df):

    print("\nDetecting Outliers")

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_monthly_spend",
        "service_dependency_index",
        "contract_risk"
    ]

    for col in numeric_cols:

        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(col,"→",len(outliers),"outliers")


# ---------------------------------------------------
# 6 REMOVE OUTLIERS
# ---------------------------------------------------

def remove_outliers(df):

    print("\nRemoving Outliers")

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_monthly_spend",
        "service_dependency_index",
        "contract_risk"
    ]

    for col in numeric_cols:

        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    print("Dataset shape:",df.shape)

    return df


# ---------------------------------------------------
# 7 REMOVE DUPLICATES
# ---------------------------------------------------

def remove_duplicates(df):

    print("\nChecking duplicates")

    duplicates = df.duplicated().sum()

    print("Duplicate rows:",duplicates)

    df = df.drop_duplicates()

    return df


# ---------------------------------------------------
# 8 CLASS IMBALANCE CHECK
# ---------------------------------------------------

def check_class_imbalance(df,target):

    print("\nChecking class imbalance")

    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True)*100)


# ---------------------------------------------------
# 9 ENCODING
# ---------------------------------------------------

def encode_categorical(df):

    print("\nOne Hot Encoding")

    df = pd.get_dummies(df, drop_first=True)

    # convert True/False to 0/1
    df = df.replace({True:1, False:0})

    return df


# ---------------------------------------------------
# 10 SPLIT FEATURES TARGET
# ---------------------------------------------------

def split_features_target(df,target):

    X = df.drop(columns=[target])
    y = df[target]

    return X,y


# ---------------------------------------------------
# 11 TRAIN TEST SPLIT
# ---------------------------------------------------

def split_train_test(X,y):

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain:",X_train.shape)
    print("Test:",X_test.shape)

    return X_train,X_test,y_train,y_test


# ---------------------------------------------------
# 12 SMOTE
# ---------------------------------------------------

def apply_smote(X_train,y_train):

    print("\nApplying SMOTE")

    smote = SMOTE(random_state=42)

    X_train,y_train = smote.fit_resample(X_train,y_train)

    print(y_train.value_counts())

    return X_train,y_train


# ---------------------------------------------------
# 13 FEATURE SCALING
# ---------------------------------------------------

def scale_features(X_train,X_test):

    print("\nFeature Scaling")

    scaler = StandardScaler()

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_monthly_spend",
        "service_dependency_index",
        "contract_risk"
    ]

    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train,X_test


# ---------------------------------------------------
# 14 FEATURE SELECTION
# ---------------------------------------------------

def feature_selection(X_train,y_train,X_test):

    print("\nFeature Selection (Forward)")

    model = LogisticRegression(max_iter=2000)

    sfs = SequentialFeatureSelector(
        model,
        direction="forward",
        n_features_to_select="auto",
        cv=5,
        n_jobs=1
    )

    sfs.fit(X_train,y_train)

    selected_features = X_train.columns[sfs.get_support()]

    print("\nSelected Features\n")
    print(selected_features)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    return X_train,X_test


# ---------------------------------------------------
# 15 SAVE DATA
# ---------------------------------------------------

def save_data(X_train,X_test,y_train,y_test):

    X_train.round(4).to_csv("processed_X_train.csv",index=False)
    X_test.round(4).to_csv("processed_X_test.csv",index=False)

    y_train.to_csv("processed_y_train.csv",index=False)
    y_test.to_csv("processed_y_test.csv",index=False)

    print("\nProcessed datasets saved")


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

def main():

    print("\nLoading dataset\n")

    df = pd.read_csv("../dataset/github_dataset.csv")

    print("Dataset shape:",df.shape)

    plot_churn_distribution(df)
    plot_churn_percentage(df)
    plot_boxplots(df)

    df = remove_identifier_columns(df)

    df = fix_datatypes(df)

    df = feature_engineering(df)

    df = handle_missing_values(df)

    detect_outliers(df)

    df = remove_outliers(df)
    plot_correlation(df)

    df = remove_duplicates(df)

    check_class_imbalance(df,"Churn")

    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    df = encode_categorical(df)

    X,y = split_features_target(df,"Churn")

    X_train,X_test,y_train,y_test = split_train_test(X,y)

    y_before = y_train.copy()

    X_train,y_train = apply_smote(X_train,y_train)

    plot_smote_balance(y_before, y_train)

    X_train,X_test = scale_features(X_train,X_test)

    X_train,X_test = feature_selection(X_train,y_train,X_test)

    print("\nFinal Shapes")
    print("Train:",X_train.shape)
    print("Test:",X_test.shape)

    save_data(X_train,X_test,y_train,y_test)


if __name__ == "__main__":
    main()