import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Income Tax Calculator", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        .stApp {background-color: #f5f5f5;}
        .title {color: #00274d; font-size: 35px; font-weight: bold;}
        .metric-label {font-size: 18px; font-weight: bold; color: #333;}
        .metric-value {font-size: 22px; font-weight: bold; color: #00274d;}
        .dataframe {font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Display Tax Summary with an Attractive Title
st.markdown(
    """
    <h2 style="text-align:center; color:#00274d; font-size:40px; font-weight:bold;">
        💰 Income Tax Calculator: Your Personalized Tax Summary 📊
    </h2>
    <hr style="border: 2px solid #00274d;">
    """,
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header("Enter Your Details")
income = st.sidebar.number_input("Enter Annual Income (₹)", min_value=0, value=1000000, step=50000)
hra_received = st.sidebar.number_input("Enter HRA Received (₹)", min_value=0, value=200000, step=5000)
rent_paid = st.sidebar.number_input("Enter Rent Paid (₹)", min_value=0, value=150000, step=5000)
basic_salary = st.sidebar.number_input("Enter Basic Salary (₹)", min_value=0, value=500000, step=50000)
is_metro = st.sidebar.checkbox("Do you live in a Metro City?", value=False)

# Deduction Inputs
deductions_80C = st.sidebar.number_input("Enter Section 80C Deductions (₹)", min_value=0, value=150000, step=50000)
deductions_other = st.sidebar.number_input("Enter Other Deductions (₹)", min_value=0, value=50000, step=50000)

# Senior Citizen Checkbox
is_senior = st.sidebar.checkbox("Are you a Senior Citizen? (60+ years)", value=False)

# Interest Income Inputs
savings_interest = st.sidebar.number_input("Savings Account Interest (₹)", min_value=0, value=5000, step=500)
fd_interest = st.sidebar.number_input("Fixed Deposit Interest (₹)", min_value=0, value=10000, step=500)

# HRA Exemption Calculation
hra_50_40 = 0.5 * basic_salary if is_metro else 0.4 * basic_salary
hra_actual = hra_received
hra_rent_based = rent_paid - (0.1 * basic_salary)
hra_exemption = max(min(hra_actual, hra_50_40, hra_rent_based), 0)

# Deduction Calculation (80TTA / 80TTB)
if is_senior:
    interest_deduction = min(savings_interest + fd_interest, 25000)  # 80TTB for Senior Citizens
    deduction_label = "80TTB Deduction (Senior Citizen)"
else:
    interest_deduction = min(savings_interest, 10000)  # 80TTA for Non-Seniors (Only Savings Interest)
    deduction_label = "80TTA Deduction (Non-Senior)"

# Total Deductions
total_deductions = deductions_80C + deductions_other + interest_deduction + hra_exemption

# Taxable Income Calculation
taxable_income = max(income - total_deductions, 0)

# Tax Calculation (Example: Flat 25% Tax Rate)
tax = taxable_income * 0.25  
cess = tax * 0.04  # 4% Cess
total_tax = tax + cess

# Tax Paid Percentage
tax_paid_percentage = (total_tax / income) * 100 if income > 0 else 0

# Display Tax Summary
st.markdown('<p class="title">📊 Tax Calculation Summary</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric(label="💵 Gross Income (₹)", value=f"{income:,.0f}")
col2.metric(label="📉 Taxable Income (₹)", value=f"{taxable_income:,.0f}")
col3.metric(label="💰 Total Tax (₹)", value=f"{total_tax:,.0f}")

# Tax Breakdown Table
data = {
    "Category": [
        "Annual Income", "HRA Exemption", "80C Deductions", "Other Deductions", deduction_label, "Total Deductions",
        "Taxable Income", "Tax (Before Cess)", "4% Cess", "Total Tax Payable", "Tax Paid (%)"
    ],
    "Amount (₹)": [
        f"{income:,.0f}", f"{hra_exemption:,.0f}", f"{deductions_80C:,.0f}", f"{deductions_other:,.0f}", f"{interest_deduction:,.0f}", f"{total_deductions:,.0f}",
        f"{taxable_income:,.0f}", f"{tax:,.0f}", f"{cess:,.0f}", f"{total_tax:,.0f}", f"{tax_paid_percentage:.2f}"
    ],
}
df = pd.DataFrame(data)

st.subheader("📋 Tax Breakdown")
st.dataframe(df.style.set_properties(**{'font-size': '18px'}))
