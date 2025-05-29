import streamlit as st
from supabase import create_client, Client

def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

def run_query():
    return supabase.table("nifty").select("*").limit(10).execute()


rows = run_query()

# Print results.
for row in rows.data:
    st.write(f"{row['date']} {row['open']}:")