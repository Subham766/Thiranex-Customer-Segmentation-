import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 38px;
    font-weight: 700;
    color: #102a43;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #2563eb;
}

.card-title {
    color: #64748b;
    font-size: 15px;
}

.card-value {
    color: #102a43;
    font-size: 28px;
    font-weight: 700;
}

.section {
    font-size: 23px;
    font-weight: 650;
    color: #102a43;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="title">👥 Customer Segmentation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze customer behavior, demographics and purchasing patterns'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Data",
    type=["csv", "xlsx"]
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = None

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

else:

    csv_path = (
        Path(__file__).resolve().parent
        / "customer_segmentation_data.csv"
    )

    if csv_path.exists():
        df = pd.read_csv(csv_path)

# ---------------------------------------------------------
# DATA CHECK
# ---------------------------------------------------------

if df is None:

    st.warning("⚠️ Customer dataset not found.")

    st.info(
        "Upload a CSV/Excel file from the sidebar or place "
        "customer_segmentation_data.csv beside app.py."
    )

    st.stop()

# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------

st.markdown(
    '<div class="section">📋 Customer Data</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(10),
    use_container_width=True
)

# ---------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "customerid",
    "age",
    "gender",
    "annualincome",
    "spendingscore",
    "purchasefrequency",
    "totalspend",
    "loyaltyyears"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        f"Missing columns: {', '.join(missing)}"
    )

    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.subheader("🔎 Filters")

filtered_df = df.copy()

gender_options = sorted(
    filtered_df["gender"].dropna().unique()
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

if selected_gender:

    filtered_df = filtered_df[
        filtered_df["gender"].isin(selected_gender)
    ]

# Age filter

min_age = int(df["age"].min())
max_age = int(df["age"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_age,
    max_age,
    (min_age, max_age)
)

filtered_df = filtered_df[
    filtered_df["age"].between(
        age_range[0],
        age_range[1]
    )
]

# Income filter

min_income = int(df["annualincome"].min())
max_income = int(df["annualincome"].max())

income_range = st.sidebar.slider(
    "Annual Income",
    min_income,
    max_income,
    (min_income, max_income)
)

filtered_df = filtered_df[
    filtered_df["annualincome"].between(
        income_range[0],
        income_range[1]
    )
]

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------

total_customers = len(filtered_df)

avg_income = filtered_df["annualincome"].mean()

avg_spending = filtered_df["spendingscore"].mean()

avg_purchase_frequency = (
    filtered_df["purchasefrequency"].mean()
)

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

st.markdown(
    '<div class="section">📊 Key Performance Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Customers</div>
            <div class="card-value">{total_customers}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Average Income</div>
            <div class="card-value">₹{avg_income:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Average Spending Score</div>
            <div class="card-value">{avg_spending:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Avg Purchase Frequency</div>
            <div class="card-value">
                {avg_purchase_frequency:.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------

st.markdown(
    '<div class="section">🤖 Customer Clustering</div>',
    unsafe_allow_html=True
)

cluster_features = [
    "age",
    "annualincome",
    "spendingscore",
    "purchasefrequency",
    "totalspend",
    "loyaltyyears"
]

cluster_data = filtered_df[cluster_features].dropna()

if len(cluster_data) < 4:

    st.warning(
        "Not enough customers for clustering."
    )

    st.stop()

# Number of clusters

num_clusters = st.sidebar.slider(
    "Number of Customer Segments",
    min_value=2,
    max_value=6,
    value=4
)

# Standardization

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    cluster_data
)

# K-Means

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    scaled_data
)

clustered_df = cluster_data.copy()

clustered_df["segment"] = (
    cluster_labels + 1
)

# Add customer ID

clustered_df["customerid"] = filtered_df.loc[
    cluster_data.index,
    "customerid"
]

# ---------------------------------------------------------
# SEGMENT DISTRIBUTION
# ---------------------------------------------------------

segment_counts = (
    clustered_df["segment"]
    .value_counts()
    .sort_index()
    .reset_index()
)

segment_counts.columns = [
    "segment",
    "customers"
]

fig = px.bar(
    segment_counts,
    x="segment",
    y="customers",
    title="Customers in Each Segment",
    text="customers"
)

fig.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Number of Customers"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# CUSTOMER SEGMENT VISUALIZATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section">🎯 Customer Segments</div>',
    unsafe_allow_html=True
)

fig = px.scatter(
    clustered_df,
    x="annualincome",
    y="spendingscore",
    color="segment",
    size="totalspend",
    hover_data=[
        "customerid",
        "age",
        "purchasefrequency",
        "loyaltyyears"
    ],
    title="Customer Segmentation by Income and Spending"
)

fig.update_layout(
    xaxis_title="Annual Income",
    yaxis_title="Spending Score"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# SEGMENT CHARACTERISTICS
# ---------------------------------------------------------

st.markdown(
    '<div class="section">📌 Segment Characteristics</div>',
    unsafe_allow_html=True
)

segment_summary = (
    clustered_df
    .groupby("segment")
    .agg(
        Customers=("customerid", "count"),
        Avg_Age=("age", "mean"),
        Avg_Income=("annualincome", "mean"),
        Avg_Spending=("spendingscore", "mean"),
        Avg_Purchases=("purchasefrequency", "mean"),
        Avg_Total_Spend=("totalspend", "mean"),
        Avg_Loyalty=("loyaltyyears", "mean")
    )
    .round(2)
    .reset_index()
)

st.dataframe(
    segment_summary,
    use_container_width=True
)

# ---------------------------------------------------------
# SEGMENT PROFILE
# ---------------------------------------------------------

st.markdown(
    '<div class="section">💡 Customer Segment Insights</div>',
    unsafe_allow_html=True
)

for _, row in segment_summary.iterrows():

    segment = int(row["segment"])

    if (
        row["Avg_Spending"] >= 70
        and row["Avg_Income"] >= df["annualincome"].median()
    ):

        description = (
            "High-value customers with strong income "
            "and high spending behavior. "
            "They are ideal for premium offers and loyalty rewards."
        )

    elif row["Avg_Spending"] >= 60:

        description = (
            "Active customers with strong purchasing behavior. "
            "Target them with personalized promotions "
            "and cross-selling campaigns."
        )

    elif row["Avg_Spending"] < 35:

        description = (
            "Low-engagement customers with relatively low "
            "spending. Consider discounts, recommendations "
            "and re-engagement campaigns."
        )

    else:

        description = (
            "Moderate-value customers who may respond well "
            "to targeted offers and product recommendations."
        )

    st.info(
        f"**Segment {segment}:** {description}"
    )

# ---------------------------------------------------------
# GENDER ANALYSIS
# ---------------------------------------------------------

st.markdown(
    '<div class="section">👤 Demographic Analysis</div>',
    unsafe_allow_html=True
)

gender_summary = (
    filtered_df["gender"]
    .value_counts()
    .reset_index()
)

gender_summary.columns = [
    "gender",
    "customers"
]

fig = px.pie(
    gender_summary,
    names="gender",
    values="customers",
    title="Customer Distribution by Gender"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# DOWNLOAD SEGMENTED DATA
# ---------------------------------------------------------

st.markdown(
    '<div class="section">⬇️ Export Segmented Data</div>',
    unsafe_allow_html=True
)

download_data = clustered_df.to_csv(
    index=False
)

st.download_button(
    label="Download Customer Segments",
    data=download_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Customer Segmentation Project | "
    "Python • Pandas • Scikit-learn • Streamlit • Plotly"
)