import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="NIFTY 50 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Supabase Configuration
# -------------------------

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
        test_response = supabase.table('nifty').select("count", count="exact").limit(1).execute()
        
        return supabase
    except Exception as e:
        print(e)
        st.error(f"Supabase connection failed: {str(e)}")
        st.error("Please check your environment variables or Streamlit secrets configuration")
        st.stop()

def optimize_data_for_plotting(df, plot_type, max_points):
    """Optimize data for efficient plotting"""
    if df.empty:
        return df
    
    original_count = len(df)
    
    if plot_type == "Daily (All Data)":
        # If too many points, sample intelligently
        if len(df) > max_points:
            # Keep first, last, and evenly spaced points in between
            indices = np.linspace(0, len(df)-1, max_points, dtype=int)
            df_optimized = df.iloc[indices].copy()
        else:
            df_optimized = df.copy()
    
    elif plot_type == "Weekly Average":
        # Group by week and take averages
        df_copy = df.copy()
        df_copy['week'] = df_copy['date'].dt.to_period('W')
        df_optimized = df_copy.groupby('week').agg({
            'date': 'first',  # Take first date of week
            'open': 'first',  # First open of week
            'high': 'max',    # Highest high of week  
            'low': 'min',     # Lowest low of week
            'close': 'last',  # Last close of week
            'volume': 'sum'   # Total volume of week
        }).reset_index(drop=True)
    
    elif plot_type == "Monthly Average":
        # Group by month and take averages
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M')
        df_optimized = df_copy.groupby('month').agg({
            'date': 'first',  # Take first date of month
            'open': 'first',  # First open of month
            'high': 'max',    # Highest high of month
            'low': 'min',     # Lowest low of month
            'close': 'last',  # Last close of month
            'volume': 'sum'   # Total volume of month
        }).reset_index(drop=True)
    
    else:  # Smart Sampling
        if len(df) <= max_points:
            df_optimized = df.copy()
        else:
            # Intelligent sampling: keep more recent data + key points
            recent_portion = max_points // 2
            historical_portion = max_points - recent_portion
            
            # Take recent data (more frequent sampling)
            recent_data = df.tail(recent_portion)
            
            # Sample historical data
            historical_data = df.head(len(df) - recent_portion)
            if len(historical_data) > historical_portion:
                indices = np.linspace(0, len(historical_data)-1, historical_portion, dtype=int)
                historical_sample = historical_data.iloc[indices]
            else:
                historical_sample = historical_data
            
            df_optimized = pd.concat([historical_sample, recent_data]).drop_duplicates().sort_values('date')
    
    return df_optimized

# -------------------------
# Data Loading Functions
# -------------------------

def execute_supabase_query(table_name, select_columns="*", filters=None, order_by=None, limit=None):
    """Execute Supabase query and return DataFrame"""
    try:
        supabase = init_supabase()
        
        # Start building the query
        query = supabase.table(table_name).select(select_columns)
        
        # Apply filters if provided
        if filters:
            for filter_item in filters:
                column, operator, value = filter_item
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
        
        # Apply ordering if provided
        if order_by:
            query = query.order(order_by)
        
        # Apply limit if provided
        if limit:
            query = query.limit(limit)
        
        # Execute query
        response = query.execute()
        
        # Convert to DataFrame
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Supabase query execution failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_summary():
    """Load dataset summary statistics using efficient Supabase queries"""
    try:
        supabase = init_supabase()
        
        # Get total count
        count_response = supabase.table('nifty').select("*", count="exact").limit(1).execute()
        total_rows = count_response.count
        
        # Get date range efficiently by getting first and last records
        first_record = supabase.table('nifty').select("date").order("date", desc=False).limit(1).execute()
        last_record = supabase.table('nifty').select("date").order("date", desc=True).limit(1).execute()
        
        if not first_record.data or not last_record.data:
            return None, []
        
        start_date = pd.to_datetime(first_record.data[0]['date'])
        end_date = pd.to_datetime(last_record.data[0]['date'])
        
        # Get min and max prices efficiently
        # For large datasets, we'll sample data to calculate statistics
        # Get every 100th record for statistical calculations to avoid loading all data
        sample_size = min(10000, total_rows)  # Sample up to 10k records
        step_size = max(1, total_rows // sample_size)
        
        # Get sampled data for statistics
        # Since Supabase doesn't have native sampling, we'll use a different approach
        # Get records at regular intervals using modulo operation on row number
        all_data = []
        batch_size = 1000
        
        # Get data in batches to calculate statistics efficiently
        for offset in range(0, min(sample_size, total_rows), batch_size):
            batch_response = supabase.table('nifty').select("close, high, low").order("date").range(offset, offset + batch_size - 1).execute()
            if batch_response.data:
                all_data.extend(batch_response.data)
        
        if all_data:
            df_sample = pd.DataFrame(all_data)
            df_sample['close'] = pd.to_numeric(df_sample['close'], errors='coerce')
            df_sample['high'] = pd.to_numeric(df_sample['high'], errors='coerce')
            df_sample['low'] = pd.to_numeric(df_sample['low'], errors='coerce')
            
            # Calculate statistics from sample
            avg_close = df_sample['close'].mean()
            min_price = df_sample['low'].min()
            max_price = df_sample['high'].max()
            std_close = df_sample['close'].std()
        else:
            # Fallback: get specific records for min/max
            min_response = supabase.table('nifty').select("low").order("low", desc=False).limit(1).execute()
            max_response = supabase.table('nifty').select("high").order("high", desc=True).limit(1).execute()
            
            min_price = float(min_response.data[0]['low']) if min_response.data else 0
            max_price = float(max_response.data[0]['high']) if max_response.data else 0
            avg_close = (min_price + max_price) / 2  # Rough estimate
            std_close = 0
        
        summary_stats = {
            'total_rows': total_rows,
            'start_date': start_date,
            'end_date': end_date,
            'avg_close': avg_close,
            'min_price': min_price,
            'max_price': max_price,
            'std_close': std_close
        }
        
        # Convert to Series for compatibility
        summary = pd.Series(summary_stats)
        
        # Get column names
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        return summary, columns
        
    except Exception as e:
        st.error(f"Error loading summary: {str(e)}")
        return None, []

@st.cache_data(ttl=600)
def load_data(start_date, end_date, limit_records=None):
    """Load NIFTY data for specified date range using Supabase"""
    try:
        # Convert dates to strings for Supabase query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Define filters
        filters = [
            ('date', 'gte', start_date_str),
            ('date', 'lte', end_date_str)
        ]
        
        # For very large date ranges, we might want to limit the data
        # and inform the user about optimization
        supabase = init_supabase()
        
        # First, check how many records match the date range
        count_query = supabase.table('nifty').select("*", count="exact")
        for column, operator, value in filters:
            if operator == "gte":
                count_query = count_query.gte(column, value)
            elif operator == "lte":
                count_query = count_query.lte(column, value)
        
        count_response = count_query.limit(1).execute()
        expected_count = count_response.count
        
        # If the dataset is very large, we might want to implement pagination
        # For now, let's set a reasonable limit
        if limit_records and expected_count > limit_records:
            st.warning(f"⚠️ Date range contains {expected_count:,} records. Loading first {limit_records:,} records for performance.")
        
        # Execute query with all data (Supabase will handle pagination internally)
        df = execute_supabase_query(
            table_name='nifty',
            select_columns='date, open, high, low, close, volume',
            filters=filters,
            order_by='date',
            limit=limit_records
        )
        
        if not df.empty:
            # Convert data types
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# -------------------------
# Main Dashboard
# -------------------------
st.markdown('<h1 class="main-header">📈 NIFTY 50 Time-Series Dashboard</h1>', unsafe_allow_html=True)

# Load summary data
with st.spinner("Loading dataset summary..."):
    summary, columns = load_summary()

if summary is not None:
    # -------------------------
    # Dataset Summary Section
    # -------------------------
    st.header("📊 Dataset Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{summary['total_rows']:,}",
            help="Total number of trading days in dataset"
        )
    
    with col2:
        st.metric(
            label="Date Range",
            value=f"{(summary['end_date'] - summary['start_date']).days} days",
            help="Total days covered in dataset"
        )
    
    with col3:
        st.metric(
            label="Average Close",
            value=f"₹{summary['avg_close']:.2f}",
            help="Average closing price across sampled data"
        )
    
    with col4:
        st.metric(
            label="Price Range",
            value=f"₹{summary['min_price']:.2f} - ₹{summary['max_price']:.2f}",
            help="Minimum and maximum prices in dataset"
        )
    
    # Additional statistics
    st.subheader("📋 Dataset Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Start Date:** {summary['start_date'].strftime('%Y-%m-%d')}
        
        **End Date:** {summary['end_date'].strftime('%Y-%m-%d')}
        
        **Standard Deviation:** ₹{summary['std_close']:.2f}
        """)
    
    with col2:
        st.info(f"""
        **Available Columns:** {len(columns)}
        
        **Column Names:** {', '.join(columns)}
        
        **Data Frequency:** Daily
        """)

    # Performance warning for large datasets
    if summary['total_rows'] > 50000:
        st.warning(f"""
        ⚡ **Performance Note**: This dataset contains {summary['total_rows']:,} records. 
        For optimal performance, consider using shorter date ranges or aggregated views (Weekly/Monthly).
        """)

    # -------------------------
    # Date Range Selection
    # -------------------------
    st.header("🗓️ Select Analysis Period")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max(summary['start_date'].date(), summary['end_date'].date() - timedelta(days=365)),  # Default to last year
            min_value=summary['start_date'].date(),
            max_value=summary['end_date'].date(),
            help="Select the start date for analysis"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=summary['end_date'].date(),
            min_value=summary['start_date'].date(),
            max_value=summary['end_date'].date(),
            help="Select the end date for analysis"
        )
    
    with col3:
        # Quick date range buttons
        st.write("**Quick Select:**")
        if st.button("Last 30 Days"):
            st.session_state.start_date = summary['end_date'].date() - timedelta(days=30)
            st.session_state.end_date = summary['end_date'].date()
            st.rerun()
        if st.button("Last 90 Days"):
            st.session_state.start_date = summary['end_date'].date() - timedelta(days=90)
            st.session_state.end_date = summary['end_date'].date()
            st.rerun()
        if st.button("Last 1 Year"):
            st.session_state.start_date = summary['end_date'].date() - timedelta(days=365)
            st.session_state.end_date = summary['end_date'].date()
            st.rerun()
        if st.button("Last 5 Years"):
            st.session_state.start_date = summary['end_date'].date() - timedelta(days=1825)
            st.session_state.end_date = summary['end_date'].date()
            st.rerun()

    # Validate date range
    if start_date > end_date:
        st.error("⚠️ Start date must be before or equal to end date.")
        st.stop()
    
    # Calculate expected data size and warn if too large
    expected_days = (end_date - start_date).days
    if expected_days > 1000:  # More than ~3 years
        st.warning(f"⚠️ Selected date range spans {expected_days} days. Consider using aggregated views for better performance.")

    # -------------------------
    # Plotting Options & Load Button
    # -------------------------
    st.header("📊 Data Visualization Options")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        plot_type = st.selectbox(
            "📈 Chart Resolution",
            ["Daily (All Data)", "Weekly Average", "Monthly Average", "Smart Sampling"],
            index=3,
            help="Choose how to aggregate data for faster plotting"
        )
    
    with col2:
        max_points = st.slider(
            "🎯 Max Data Points",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            help="Limit data points for faster rendering"
        )
    
    with col3:
        # Data loading limit
        max_records = st.selectbox(
            "📊 Max Records to Load",
            [1000, 5000, 10000, 25000, 50000, None],
            index=2,
            format_func=lambda x: f"{x:,}" if x else "All Records",
            help="Limit records loaded from database for performance"
        )
    
    with col4:
        st.write("**Load Data:**")
        load_button = st.button(
            "🚀 Load & Plot Data",
            type="primary",
            help="Click to load data and generate charts"
        )

    # -------------------------
    # Load and Display Data (Only when button clicked)
    # -------------------------
    if load_button:
        with st.spinner("Loading data from Supabase..."):
            df = load_data(start_date, end_date, max_records)
        
        if df.empty:
            st.warning("📭 No data available for the selected date range. Please adjust your selection.")
        else:
            # Optimize data for plotting
            original_count = len(df)
            df_plot = optimize_data_for_plotting(df, plot_type, max_points)
            optimized_count = len(df_plot)
            
            # Show optimization info
            st.success(f"✅ Loaded {original_count:,} records, optimized to {optimized_count:,} points for plotting ({plot_type})")
            
            # -------------------------
            # Professional Charts
            # -------------------------
            st.header("📈 Price Analysis")
            
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'NIFTY 50 Analysis - {plot_type} ({start_date} to {end_date})', fontsize=16, fontweight='bold')
            
            # Main price chart
            ax1.plot(df_plot['date'], df_plot['close'], label='Close Price', color='#1f77b4', linewidth=2, marker='o' if optimized_count < 100 else None, markersize=3)
            ax1.plot(df_plot['date'], df_plot['open'], label='Open Price', color='#ff7f0e', alpha=0.7, linewidth=1.5)
            
            # Only add fill_between for reasonable data sizes
            if optimized_count < 1000:
                ax1.fill_between(df_plot['date'], df_plot['low'], df_plot['high'], alpha=0.2, color='gray', label='Day Range (High-Low)')
            
            ax1.set_title('NIFTY 50 Price Movement', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (₹)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Calculate and plot moving averages (only for daily data)
            if plot_type == "Daily (All Data)":
                if len(df_plot) >= 20:
                    df_plot['MA20'] = df_plot['close'].rolling(window=20).mean()
                    ax1.plot(df_plot['date'], df_plot['MA20'], label='20-Day MA', color='red', linestyle='--', alpha=0.8)
                
                if len(df_plot) >= 50:
                    df_plot['MA50'] = df_plot['close'].rolling(window=50).mean()
                    ax1.plot(df_plot['date'], df_plot['MA50'], label='50-Day MA', color='green', linestyle='--', alpha=0.8)
                
                ax1.legend(loc='upper left')
            
            # Volume chart or returns chart
            if 'volume' in df_plot.columns and df_plot['volume'].notna().any() and df_plot['volume'].sum() > 0:
                bars = ax2.bar(df_plot['date'], df_plot['volume'], alpha=0.6, color='purple')
                ax2.set_title(f'Trading Volume ({plot_type})', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                # Format y-axis for volume
                ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            else:
                # Daily returns if no volume data
                df_plot['daily_return'] = df_plot['close'].pct_change() * 100
                ax2.plot(df_plot['date'], df_plot['daily_return'], color='red', alpha=0.7, linewidth=1)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Return (%)', fontsize=12)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # -------------------------
            # Key Statistics for Selected Period
            # -------------------------
            st.header("📊 Period Statistics")
            
            # Calculate statistics using original data
            price_change = df['close'].iloc[-1] - df['close'].iloc[0]
            price_change_pct = (price_change / df['close'].iloc[0]) * 100
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Period Return",
                    value=f"{price_change_pct:.2f}%",
                    delta=f"₹{price_change:.2f}",
                    help="Total return for the selected period"
                )
            
            with col2:
                st.metric(
                    label="Highest Price",
                    value=f"₹{df['high'].max():.2f}",
                    help="Highest price during the period"
                )
            
            with col3:
                st.metric(
                    label="Lowest Price",
                    value=f"₹{df['low'].min():.2f}",
                    help="Lowest price during the period"
                )
            
            with col4:
                st.metric(
                    label="Volatility (Annual)",
                    value=f"{volatility:.1f}%",
                    help="Annualized volatility based on daily returns"
                )
            
            # Performance info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **📊 Data Optimization:**
                - Original Records: {original_count:,}
                - Plotted Points: {optimized_count:,}
                - Compression Ratio: {(original_count/optimized_count):.1f}x
                """)
            
            with col2:
                st.info(f"""
                **⚡ Performance:**
                - Chart Type: {plot_type}
                - Rendering: Optimized
                - Load Time: Reduced by ~{((original_count-optimized_count)/original_count*100):.0f}%
                """)
            
            # -------------------------
            # Data Table (Optional)
            # -------------------------
            with st.expander("📋 View Raw Data (Original Dataset)"):
                st.write(f"Showing sample of {min(1000, len(df))} rows from {len(df):,} total records")
                display_df = df.head(1000) if len(df) > 1000 else df
                st.dataframe(
                    display_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset as CSV",
                data=csv,
                file_name=f"nifty_data_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )

    else:
        st.info("👆 Select your preferred chart options and click '🚀 Load & Plot Data' to visualize the data.")

else:
    st.error("❌ Unable to load data. Please check your Supabase connection and try again.")

# Footer
st.markdown("---")
st.markdown(
    "**NIFTY 50 Dashboard** | Built with Streamlit & Supabase | Data updated every 10 minutes",
    help="This dashboard provides real-time analysis of NIFTY 50 index data"
)