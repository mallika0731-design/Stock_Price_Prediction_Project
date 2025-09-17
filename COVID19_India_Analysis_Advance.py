# COVID19_India_Analysis_Advance.py
# Advanced COVID-19 Analysis and Dashboard for India using Streamlit and Plotly 
# Data Source: https://data.incovid19.org/csv/latest/
# COVID-19 India Analysis + Streamlit Dashboard
# COVID-19 India Analysis + Streamlit Dashboard

# COVID-19 India Analysis + Streamlit Dashboard
import os
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------
# STREAMLIT PAGE CONFIG MUST BE FIRST
# -------------------------
st.set_page_config(page_title="COVID-19 India Analysis", layout="wide")

# -------------------------
# Desktop data folder
# -------------------------
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
data_dir = os.path.join(desktop_dir, "data")

# Verify folder exists
if not os.path.exists(data_dir):
    st.error("❌ 'data' folder not found on Desktop. Please place your CSVs there.")
    st.stop()

# -------------------------
# Map actual filenames
# -------------------------
states_file = os.path.join(data_dir, "states_cases.csv")
districts_file = os.path.join(data_dir, "districts_cases.csv")
vaccination_file = os.path.join(data_dir, "vaccination_statewise.csv")

# Verify files exist
for f, n in zip([states_file, districts_file, vaccination_file], ["States", "Districts", "Vaccination"]):
    if not os.path.exists(f):
        st.error(f"❌ {n} CSV file not found: {f}")
        st.stop()

# -------------------------
# Load datasets
# -------------------------
@st.cache_data
def load_data():
    states_df = pd.read_csv(states_file)
    districts_df = pd.read_csv(districts_file)
    vacc_df = pd.read_csv(vaccination_file)

    # Convert date columns if present
    for df, col in [(states_df, "Date"), (districts_df, "Date"), (vacc_df, "Updated On")]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return states_df, vacc_df, districts_df

states_df, vacc_df, districts_df = load_data()

# -------------------------
# Streamlit title and description
# -------------------------
st.title("📊 COVID-19 India Analysis & Dashboard")
st.markdown("Interactive dashboard with cases, vaccinations, and district-level data.")

# -------------------------
# 1. National Overview
# -------------------------
st.header("1. National Overview")
latest_date = states_df["Date"].max()
latest_data = states_df[states_df["Date"] == latest_date]

total_confirmed = int(latest_data["Confirmed"].sum())
total_recovered = int(latest_data["Recovered"].sum())
total_deceased = int(latest_data["Deceased"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Confirmed", f"{total_confirmed:,}")
col2.metric("Total Recovered", f"{total_recovered:,}")
col3.metric("Total Deceased", f"{total_deceased:,}")

# -------------------------
# 2. State-wise cases
# -------------------------
st.header("2. State-wise COVID-19 Cases")
state_cases = latest_data.groupby("State")[["Confirmed", "Recovered", "Deceased"]].sum().reset_index()
fig_state = px.bar(state_cases.sort_values("Confirmed", ascending=False),
                   x="Confirmed", y="State", orientation="h",
                   title="State-wise Confirmed Cases", height=700)
st.plotly_chart(fig_state, use_container_width=True)

# -------------------------
# 3. Trend over time
# -------------------------
st.header("3. Cases Trend Over Time")
trend = states_df.groupby("Date")[["Confirmed", "Recovered", "Deceased"]].sum().reset_index()
fig_trend = px.line(trend, x="Date", y=["Confirmed", "Recovered", "Deceased"],
                    title="National Trend Over Time")
st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# 4. Active vs Recovered
# -------------------------
st.header("4. Active vs Recovered Cases")
latest_data["Active"] = latest_data["Confirmed"] - latest_data["Recovered"] - latest_data["Deceased"]
active_data = latest_data.groupby("State")[["Active", "Recovered"]].sum().reset_index()
fig_active = px.bar(active_data.melt(id_vars="State", value_vars=["Active", "Recovered"]),
                    x="value", y="State", color="variable", orientation="h",
                    title="Active vs Recovered Cases by State", height=700)
st.plotly_chart(fig_active, use_container_width=True)

# -------------------------
# 5. Case Fatality Rate (CFR)
# -------------------------
st.header("5. Case Fatality Rate by State")
cfr = (latest_data.groupby("State")["Deceased"].sum() /
       latest_data.groupby("State")["Confirmed"].sum() * 100).reset_index()
cfr.columns = ["State", "CFR (%)"]
fig_cfr = px.bar(cfr.sort_values("CFR (%)", ascending=False),
                 x="CFR (%)", y="State", orientation="h",
                 title="Case Fatality Rate (%) by State", height=700)
st.plotly_chart(fig_cfr, use_container_width=True)

# -------------------------
# 6. Vaccination progress
# -------------------------
st.header("6. Vaccination Progress")
latest_vacc = vacc_df[vacc_df["Updated On"] == vacc_df["Updated On"].max()]
vacc_summary = latest_vacc.groupby("State")[["Total Doses Administered"]].sum().reset_index()
fig_vacc = px.bar(vacc_summary.sort_values("Total Doses Administered", ascending=False),
                  x="Total Doses Administered", y="State", orientation="h",
                  title="State-wise Vaccination Progress", height=700)
st.plotly_chart(fig_vacc, use_container_width=True)

# -------------------------
# 7. Vaccination Trend Over Time
# -------------------------
st.header("7. Vaccination Trend Over Time")
vacc_trend = vacc_df.groupby("Updated On")[["Total Doses Administered"]].sum().reset_index()
fig_vacc_trend = px.line(vacc_trend, x="Updated On", y="Total Doses Administered",
                         title="National Vaccination Trend Over Time")
st.plotly_chart(fig_vacc_trend, use_container_width=True)

# -------------------------
# 8. District-level analysis
# -------------------------
st.header("8. District-level Analysis")
selected_state = st.selectbox("Select a State", districts_df["State"].unique())
district_data = districts_df[districts_df["State"] == selected_state]
latest_district = district_data[district_data["Date"] == district_data["Date"].max()]
fig_district = px.bar(latest_district.sort_values("Confirmed", ascending=False),
                      x="Confirmed", y="District", orientation="h",
                      title=f"District-wise Cases in {selected_state}", height=700)
st.plotly_chart(fig_district, use_container_width=True)

# -------------------------
# 9. Growth Rate of Cases
# -------------------------
st.header("9. Growth Rate of Cases")
trend["Daily Growth"] = trend["Confirmed"].diff().fillna(0)
fig_growth = px.line(trend, x="Date", y="Daily Growth", title="Daily Growth of Cases in India")
st.plotly_chart(fig_growth, use_container_width=True)

# -------------------------
# 10. Recovery & Mortality Trends
# -------------------------
st.header("10. Recovery & Mortality Trends")
trend["Recovery Rate (%)"] = (trend["Recovered"] / trend["Confirmed"] * 100).fillna(0)
trend["Mortality Rate (%)"] = (trend["Deceased"] / trend["Confirmed"] * 100).fillna(0)
fig_rates = px.line(trend, x="Date", y=["Recovery Rate (%)", "Mortality Rate (%)"],
                    title="Recovery & Mortality Rates Over Time")
st.plotly_chart(fig_rates, use_container_width=True)

# -------------------------
# 11. Top States Comparison
# -------------------------
st.header("11. Top States Comparison")
top_states = state_cases.sort_values("Confirmed", ascending=False).head(10)
fig_top = px.bar(top_states.melt(id_vars="State", value_vars=["Confirmed", "Recovered", "Deceased"]),
                 x="State", y="value", color="variable", barmode="group",
                 title="Top 10 States by Cases Comparison")
st.plotly_chart(fig_top, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("✅ Data Source: [IN COVID19 Data API](https://data.incovid19.org/csv/latest/)")
st.markdown("Built with ❤️ using **Streamlit + Plotly**")
