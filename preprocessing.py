import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# giới thiệu dữ liệu
df = pd.read_csv("data/diabetes.csv")
print(df.head(10))
print(df["Outcome"].value_counts())
print(df.shape)
print(df.info())
print(df.describe())
print(df.groupby("Outcome").mean())

# tiền xử lý
def dienkhuyetthieu(df):
    # Replacing NaN with mean values
    df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = df[
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
    df["Glucose"].fillna(df["Glucose"].mean(), inplace=True)
    df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
    df["SkinThickness"].fillna(df["SkinThickness"].mean(), inplace=True)
    df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)
    df["BMI"].fillna(df["BMI"].mean(), inplace=True)



# chuẩn hóa dữ liệu
def chuanhoa(X):
    X = df.drop(columns="Outcome", axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    return standardized_data



