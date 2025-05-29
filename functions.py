import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

@st.cache_resource
def init_supabase() -> Client:
    """Initialize Supabase client"""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and Key must be provided")
        supabase: Client = create_client(supabase_url, supabase_key)
        # Test connection
        supabase.table('nifty').select("count", count="exact").limit(1).execute()
        return supabase
    except Exception as e:
        st.error(f"Supabase connection failed: {str(e)}")
        st.stop()

def optimize_data_for_plotting(df, plot_type, max_points):
    if df.empty:
        return df
    if plot_type == "Daily (All Data)":
        if len(df) > max_points:
            indices = np.linspace(0, len(df)-1, max_points, dtype=int)
            return df.iloc[indices].copy()
        else:
            return df.copy()
    elif plot_type == "Weekly Average":
        df_copy = df.copy()
        df_copy['week'] = df_copy['date'].dt.to_period('W')
        return df_copy.groupby('week').agg({
            'date': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).reset_index(drop=True)
    elif plot_type == "Monthly Average":
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M')
        return df_copy.groupby('month').agg({
            'date': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).reset_index(drop=True)
    else:  # Smart Sampling
        if len(df) <= max_points:
            return df.copy()
        else:
            recent_portion = max_points // 2
            historical_portion = max_points - recent_portion
            recent_data = df.tail(recent_portion)
            historical_data = df.head(len(df) - recent_portion)
            if len(historical_data) > historical_portion:
                indices = np.linspace(0, len(historical_data)-1, historical_portion, dtype=int)
                historical_sample = historical_data.iloc[indices]
            else:
                historical_sample = historical_data
            return pd.concat([historical_sample, recent_data]).drop_duplicates().sort_values('date')

def execute_supabase_query(table_name, select_columns="*", filters=None, order_by=None, limit=None):
    try:
        supabase = init_supabase()
        query = supabase.table(table_name).select(select_columns)
        if filters:
            for column, operator, value in filters:
                if operator == "gte":
                    query = query.gte(column, value)
                elif operator == "lte":
                    query = query.lte(column, value)
                elif operator == "eq":
                    query = query.eq(column, value)
                elif operator == "gt":
                    query = query.gt(column, value)
                elif operator == "lt":
                    query = query.lt(column, value)
        if order_by:
            query = query.order(order_by)
        if limit:
            query = query.limit(limit)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Supabase query execution failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_summary():
    try:
        supabase = init_supabase()
        count_response = supabase.table('nifty').select("*", count="exact").limit(1).execute()
        total_rows = count_response.count
        first_record = supabase.table('nifty').select("date").order("date", desc=False).limit(1).execute()
        last_record = supabase.table('nifty').select("date").order("date", desc=True).limit(1).execute()
        if not first_record.data or not last_record.data:
            return None, []
        start_date = pd.to_datetime(first_record.data[0]['date'])
        end_date = pd.to_datetime(last_record.data[0]['date'])
        sample_size = min(10000, total_rows)
        all_data = []
        batch_size = 1000
        for offset in range(0, min(sample_size, total_rows), batch_size):
            batch_response = supabase.table('nifty').select("close, high, low").order("date").range(offset, offset + batch_size - 1).execute()
            if batch_response.data:
                all_data.extend(batch_response.data)
        if all_data:
            df_sample = pd.DataFrame(all_data)
            df_sample['close'] = pd.to_numeric(df_sample['close'], errors='coerce')
            df_sample['high'] = pd.to_numeric(df_sample['high'], errors='coerce')
            df_sample['low'] = pd.to_numeric(df_sample['low'], errors='coerce')
            avg_close = df_sample['close'].mean()
            min_price = df_sample['low'].min()
            max_price = df_sample['high'].max()
            std_close = df_sample['close'].std()
        else:
            min_price = max_price = avg_close = std_close = 0
        summary_stats = {
            'total_rows': total_rows,
            'start_date': start_date,
            'end_date': end_date,
            'avg_close': avg_close,
            'min_price': min_price,
            'max_price': max_price,
            'std_close': std_close
        }
        summary = pd.Series(summary_stats)
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        return summary, columns
    except Exception as e:
        st.error(f"Error loading summary: {str(e)}")
        return None, []

@st.cache_data(ttl=600)
def load_data(start_date, end_date, limit_records=None):
    try:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        filters = [
            ('date', 'gte', start_date_str),
            ('date', 'lte', end_date_str)
        ]
        supabase = init_supabase()
        count_query = supabase.table('nifty').select("*", count="exact")
        for column, operator, value in filters:
            if operator == "gte":
                count_query = count_query.gte(column, value)
            elif operator == "lte":
                count_query = count_query.lte(column, value)
        count_response = count_query.limit(1).execute()
        expected_count = count_response.count
        if limit_records and expected_count > limit_records:
            st.warning(f"⚠️ Date range contains {expected_count:,} records. Loading first {limit_records:,} records for performance.")
        df = execute_supabase_query(
            table_name='nifty',
            select_columns='date, open, high, low, close, volume',
            filters=filters,
            order_by='date',
            limit=limit_records
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()
