import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# === Load and prepare data ===
df = pd.read_csv("Job.csv")
required_columns = ["Type_of_job", "company_name", "experience", "salary"]
df = df.dropna(subset=required_columns)

def clean_salary(s):
    s = re.sub(r"[^\d.]", "", str(s))
    s = re.sub(r"\.(?=.*\.)", "", s)
    return s

df["salary_cleaned"] = df["salary"].apply(clean_salary)
df = df[df["salary_cleaned"].str.match(r"^\d+(\.\d+)?$")]
df["salary_numeric"] = df["salary_cleaned"].astype(float)

# Encode features
le_job = LabelEncoder()
le_company = LabelEncoder()
le_exp = LabelEncoder()

df["Type_of_job_enc"] = le_job.fit_transform(df["Type_of_job"])
df["company_name_enc"] = le_company.fit_transform(df["company_name"])
df["experience_enc"] = le_exp.fit_transform(df["experience"])

X = df[["Type_of_job_enc", "company_name_enc", "experience_enc"]]
y = df["salary_numeric"]

model = LinearRegression()
model.fit(X, y)

# === GUI ===
root = tk.Tk()
root.title("Job Salary Predictor")
root.geometry("500x400")
root.configure(bg="#f5f5f5")
root.resizable(False, False)

# === Title ===
title_label = tk.Label(
    root,
    text="Job Salary Prediction Tool",
    font=("Helvetica", 16, "bold"),
    bg="#f5f5f5",
    fg="#2c3e50"
)
title_label.pack(pady=15)

# === Job Title Dropdown ===
tk.Label(root, text="Select Job Title:", font=("Arial", 12), bg="#f5f5f5").pack()
job_var = tk.StringVar()
job_menu = ttk.Combobox(root, textvariable=job_var, values=sorted(df["Type_of_job"].unique()), state="readonly", width=40)
job_menu.pack(pady=5)

# === Company Name Dropdown ===
tk.Label(root, text="Select Company Name:", font=("Arial", 12), bg="#f5f5f5").pack()
company_var = tk.StringVar()
company_menu = ttk.Combobox(root, textvariable=company_var, values=sorted(df["company_name"].unique()), state="readonly", width=40)
company_menu.pack(pady=5)

# === Experience Level Dropdown ===
tk.Label(root, text="Select Experience Level:", font=("Arial", 12), bg="#f5f5f5").pack()
exp_var = tk.StringVar()
exp_menu = ttk.Combobox(root, textvariable=exp_var, values=sorted(df["experience"].unique()), state="readonly", width=40)
exp_menu.pack(pady=5)

# === Result Label ===
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="blue", bg="#f5f5f5")
result_label.pack(pady=20)

# === Predict Button ===
def predict_salary():
    try:
        job_enc = le_job.transform([job_var.get()])[0]
        comp_enc = le_company.transform([company_var.get()])[0]
        exp_enc = le_exp.transform([exp_var.get()])[0]

        pred = model.predict([[job_enc, comp_enc, exp_enc]])
        result_label.config(text=f"Predicted Salary: ₹ {pred[0]:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

tk.Button(root, text="Predict Salary", command=predict_salary, font=("Arial", 12), bg="#2980b9", fg="white", padx=10, pady=5).pack(pady=10)

root.mainloop()
