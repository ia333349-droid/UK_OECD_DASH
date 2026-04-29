import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="PISA 2022 ICT Analysis", layout="wide")
st.title("📊 Modern Technology in Classrooms")
st.subheader("PISA 2022 Data Analytics | University of Portsmouth MSc Project")

# Load data
@st.cache_data
def load_data():
    return pd.read_pickle("data_cleaned/stu_clean.pkl")

stu = load_data()

# Sidebar
st.sidebar.header("Filters")
view = st.sidebar.radio("View", ["United Kingdom Focus", "UK vs OECD Comparison"])
subject = st.sidebar.selectbox("Subject", ["Mathematics", "Reading", "Science"])

# Prepare data
ict_vars = ['IC170Q01JA', 'IC171Q01JA', 'IC172Q01JA', 'IC173Q01JA', 
            'IC174Q01JA', 'IC175Q01JA', 'IC176Q01JA']

# ====================== UK ANALYSIS ======================
uk = stu[stu['CNT'] == 'GBR'].copy()

cluster_data = uk[ict_vars].copy()
for col in ict_vars:
    cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())

scaler = StandardScaler()
X = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
uk['ICT_Cluster'] = kmeans.fit_predict(X)
uk['ICT_Cluster_Label'] = uk['ICT_Cluster'].map({
    0: "Low ICT Users", 
    1: "Moderate ICT Users", 
    2: "High ICT Users"
})

uk['Math_Score'] = uk['PV1MATH']
uk['Reading_Score'] = uk['PV1READ']
uk['Science_Score'] = uk['PV1SCIE']

score_col = "Math_Score" if subject == "Mathematics" else "Reading_Score" if subject == "Reading" else "Science_Score"

# ====================== MAIN DASHBOARD ======================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("ICT Usage Clusters in UK")
    fig_pie = px.pie(uk, names='ICT_Cluster_Label', title="Distribution of Students by ICT Usage")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader(f"{subject} Performance by Cluster")
    perf = uk.groupby('ICT_Cluster_Label')[score_col].mean().round(2)
    fig_bar = px.bar(perf, title=f"Average {subject} Score", text_auto=True,
                     labels={'value': 'Average Score'})
    st.plotly_chart(fig_bar, use_container_width=True)

# ====================== COMPARISON SECTION ======================
st.markdown("---")
st.subheader("UK vs OECD Average Comparison")

if view == "UK vs OECD Comparison":
    oecd = stu[stu['CNT'] != 'GBR'].copy()
    
    cluster_data_o = oecd[ict_vars].copy()
    for col in ict_vars:
        cluster_data_o[col] = cluster_data_o[col].fillna(cluster_data_o[col].median())
    
    X_o = scaler.fit_transform(cluster_data_o)
    oecd['ICT_Cluster'] = kmeans.fit_predict(X_o)
    oecd['ICT_Cluster_Label'] = oecd['ICT_Cluster'].map({0: "Low ICT Users", 1: "Moderate ICT Users", 2: "High ICT Users"})
    oecd[score_col] = oecd['PV1MATH'] if subject == "Mathematics" else oecd['PV1READ'] if subject == "Reading" else oecd['PV1SCIE']

    comp = pd.DataFrame({
        'UK': uk.groupby('ICT_Cluster_Label')[score_col].mean().round(2),
        'OECD Average': oecd.groupby('ICT_Cluster_Label')[score_col].mean().round(2)
    })

    fig_comp = px.bar(comp.reset_index().melt(id_vars='ICT_Cluster_Label'), 
                      x='ICT_Cluster_Label', y='value', color='variable',
                      barmode='group', title=f"UK vs OECD Average {subject} Scores by ICT Cluster",
                      text_auto=True)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(comp, use_container_width=True)

else:
    st.info("Switch to 'UK vs OECD Comparison' in the sidebar to see international comparison.")

st.caption("Data Source: OECD PISA 2022 | Analysis by Hariram Behera")