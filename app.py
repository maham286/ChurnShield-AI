import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# page configuration
st.set_page_config(
    page_title="ChurnShield AI — Executive Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css injected for custom theme
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
        color: #E0E6ED;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B132B;
        border-right: 1px solid rgba(0, 255, 204, 0.15);
    }
    
    /* Modern Glassmorphic KPI Cards */
    .kpi-card {
        background: rgba(28, 37, 65, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 204, 0.15);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: #00FFCC;
        box-shadow: 0 12px 40px 0 rgba(0, 255, 204, 0.15);
    }
    .kpi-title {
        color: #8D99AE;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .text-teal { color: #00FFCC; }
    .text-coral { color: #FF4B4B; }
    .text-amber { color: #FFB703; }

    /* Custom Header Banner */
    .header-banner {
        padding: 24px 32px;
        background: linear-gradient(90deg, rgba(0,255,204,0.08) 0%, rgba(28,37,65,0.4) 100%);
        border-bottom: 2px solid rgba(0, 255, 204, 0.2);
        border-radius: 16px;
        margin-bottom: 25px;
    }
    
    /* Hide Default Streamlit Menu & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# model loader
@st.cache_resource
def load_models():
    model_path = 'saved_models/xgboost_churn_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_models()

# header banner
st.markdown("""
<div class="header-banner">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="margin:0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">
                🛡️ ChurnShield <span style="color: #00FFCC;">AI</span>
            </h1>
            <p style="margin: 4px 0 0 0; color: #8D99AE; font-size: 0.95rem;">
                Enterprise Retention Intelligence & Revenue Risk Diagnostics
            </p>
        </div>
        <div style="background: rgba(0,255,204,0.1); border: 1px solid #00FFCC; padding: 6px 16px; border-radius: 20px; color: #00FFCC; font-size: 0.85rem; font-weight: 600;">
            ● Live Engine Active
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# data handling 
st.sidebar.markdown("### ⚙️ Engine Controls")
uploaded_file = st.sidebar.file_uploader("Ingest Customer Data (.csv)", type=["csv"])

@st.cache_data
def get_data(file):
    if file is not None:
        return pd.read_csv(file)
    elif os.path.exists('data/processed_churn_data.csv'):
        return pd.read_csv('data/processed_churn_data.csv')
    elif os.path.exists('data/raw_telecom_churn.csv'):
        return pd.read_csv('data/raw_telecom_churn.csv')
    return None

raw_df = get_data(uploaded_file)

if raw_df is not None:
    df = raw_df.copy()
    
    # feature engineering pipeline
    df_clean = df.copy()
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].replace(' ', np.nan), errors='coerce')
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
    
    if 'tenure' in df_clean.columns:
        df_clean['TenureGroup'] = pd.cut(
            df_clean['tenure'], 
            bins=[-1, 12, 24, 48, 72, 100], 
            labels=['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years', '6+ Years']
        )
    if 'MonthlyCharges' in df_clean.columns and 'TotalCharges' in df_clean.columns:
        df_clean['ChargeRatio'] = df_clean['TotalCharges'] / (df_clean['MonthlyCharges'] + 1e-5)
        
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    if 'customerID' in cat_cols:
        cat_cols = cat_cols.drop('customerID')
        
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    
    # model scoring pipeline
    if model is not None:
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_features]
        
        probs = model.predict_proba(df_encoded)[:, 1]
        df['Churn_Probability'] = np.round(probs, 3)
        df['Risk_Level'] = pd.cut(df['Churn_Probability'], bins=[-0.01, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
    else:
        df['Churn_Probability'] = 0.25
        df['Risk_Level'] = 'Low'

    # key metrics calculations
    total_cust = len(df)
    high_risk_df = df[df['Risk_Level'] == 'High']
    med_risk_df = df[df['Risk_Level'] == 'Medium']
    
    at_risk_mrr = high_risk_df['MonthlyCharges'].sum() if 'MonthlyCharges' in df.columns else 0
    at_risk_arr = at_risk_mrr * 12

    # KPI cards grid
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monitored Portfolio</div>
            <div class="kpi-value">{total_cust:,}</div>
            <div class="kpi-sub text-teal">Active Accounts Scored</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Critical Churn Risk</div>
            <div class="kpi-value">{len(high_risk_df):,}</div>
            <div class="kpi-sub text-coral">{(len(high_risk_df)/total_cust)*100:.1f}% High Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">At-Risk MRR Impact</div>
            <div class="kpi-value">${at_risk_mrr:,.0f}</div>
            <div class="kpi-sub text-amber">Monthly Exposure</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">At-Risk ARR Impact</div>
            <div class="kpi-value">${at_risk_arr:,.0f}</div>
            <div class="kpi-sub text-coral">Annual Revenue at Risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # tabs
    # tsabs
    tab1, tab2, tab3 = st.tabs(["📊 Risk Matrix & Visuals", "📋 At-Risk Target Roster", "⚡ Retention Simulator"])

    with tab1:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Level', 'Count']
            
            fig_donut = px.pie(
                risk_counts, 
                values='Count', 
                names='Risk_Level',
                hole=0.65,
                color='Risk_Level',
                color_discrete_map={'Low': '#00FFCC', 'Medium': '#FFB703', 'High': '#FF4B4B'},
                title="<b>Portfolio Risk Distribution</b>"
            )
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E6ED'),
                showlegend=True,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_right:
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                fig_scatter = px.scatter(
                    df, 
                    x='tenure', 
                    y='MonthlyCharges', 
                    color='Risk_Level',
                    size='Churn_Probability',
                    color_discrete_map={'Low': '#00FFCC', 'Medium': '#FFB703', 'High': '#FF4B4B'},
                    title="<b>Tenure vs Monthly Spend (Bubble Size = Churn Risk)</b>",
                    labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)'}
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E0E6ED'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.markdown("#### 🚨 Prioritized At-Risk Customer Directory")
        display_cols = [c for c in ['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'Churn_Probability', 'Risk_Level'] if c in df.columns]
        
        filter_risk = st.multiselect("Filter by Risk Level", ['High', 'Medium', 'Low'], default=['High', 'Medium'])
        filtered_df = df[df['Risk_Level'].isin(filter_risk)].sort_values(by='Churn_Probability', ascending=False)
        
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            height=380
        )

    with tab3:
        st.markdown("#### 💡 Simulated Retention ROI")
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            discount_offer = st.slider("Target Discount Campaign (%)", 5, 30, 15)
            success_rate = st.slider("Expected Campaign Success Rate (%)", 10, 60, 25)
            
            saved_customers = int(len(high_risk_df) * (success_rate / 100))
            retained_mrr = saved_customers * (high_risk_df['MonthlyCharges'].mean() if len(high_risk_df) > 0 else 0) * (1 - discount_offer/100)
            
            st.metric("Estimated Customers Saved", f"{saved_customers} Users")
            st.metric("Recovered Monthly Revenue", f"${retained_mrr:,.2f}")
            
        with col_s2:
            st.info(f"""
            **Retention Strategy Impact Summary:**
            * Targeting **{len(high_risk_df)} high-risk users** with a **{discount_offer}% price incentive**.
            * Assuming a **{success_rate}% conversion rate**, you prevent imminent churn for approximately **{saved_customers} accounts**.
            * Net Annualized Saved ARR: **${retained_mrr * 12:,.2f}**.
            """)

else:
    st.error("No dataset found in directory. Upload a CSV file using the sidebar.")