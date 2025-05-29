import streamlit as st
from nifty_dashboard import show_nifty_dashboard

# Set Streamlit page config
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
tabs = ["NIFTY 50 Dashboard"]
# You can add more tabs/pages here in the future, e.g.:
# tabs.append("Other Dashboard")

selected_tab = st.sidebar.radio("Go to", tabs)

if selected_tab == "NIFTY 50 Dashboard":
    show_nifty_dashboard()
# elif selected_tab == "Other Dashboard":
#     show_other_dashboard()
