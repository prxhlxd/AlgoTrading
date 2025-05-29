import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
from functions import load_summary, load_data, optimize_data_for_plotting

def show_nifty_dashboard():
    st.markdown('<h1 class="main-header">📈 NIFTY 50 Time-Series Dashboard</h1>', unsafe_allow_html=True)

    with st.spinner("Loading dataset summary..."):
        summary, columns = load_summary()

    if summary is not None:
        st.header("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{summary['total_rows']:,}", help="Total number of trading days in dataset")
        with col2:
            st.metric("Date Range", f"{(summary['end_date'] - summary['start_date']).days} days", help="Total days covered in dataset")
        with col3:
            st.metric("Average Close", f"₹{summary['avg_close']:.2f}", help="Average closing price across sampled data")
        with col4:
            st.metric("Price Range", f"₹{summary['min_price']:.2f} - ₹{summary['max_price']:.2f}", help="Minimum and maximum prices in dataset")
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
            **Data Frequency:** Minute-Level
            """)
        if summary['total_rows'] > 50000:
            st.warning(f"""
            ⚡ **Performance Note**: This dataset contains {summary['total_rows']:,} records. 
            For optimal performance, consider using shorter date ranges or aggregated views (Weekly/Monthly).
            """)
        st.header("🗓️ Select Analysis Period")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max(summary['start_date'].date(), summary['end_date'].date() - timedelta(days=365)),
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
        if start_date > end_date:
            st.error("⚠️ Start date must be before or equal to end date.")
            st.stop()
        expected_days = (end_date - start_date).days
        if expected_days > 1000:
            st.warning(f"⚠️ Selected date range spans {expected_days} days. Consider using aggregated views for better performance.")
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
        if load_button:
            with st.spinner("Loading data from Supabase..."):
                df = load_data(start_date, end_date, max_records)
            if df.empty:
                st.warning("📭 No data available for the selected date range. Please adjust your selection.")
            else:
                original_count = len(df)
                df_plot = optimize_data_for_plotting(df, plot_type, max_points)
                optimized_count = len(df_plot)
                st.success(f"✅ Loaded {original_count:,} records, optimized to {optimized_count:,} points for plotting ({plot_type})")
                st.header("📈 Price Analysis")
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                fig.suptitle(f'NIFTY 50 Analysis - {plot_type} ({start_date} to {end_date})', fontsize=16, fontweight='bold')
                ax1.plot(df_plot['date'], df_plot['close'], label='Close Price', color='#1f77b4', linewidth=2, marker='o' if optimized_count < 100 else None, markersize=3)
                ax1.plot(df_plot['date'], df_plot['open'], label='Open Price', color='#ff7f0e', alpha=0.7, linewidth=1.5)
                if optimized_count < 1000:
                    ax1.fill_between(df_plot['date'], df_plot['low'], df_plot['high'], alpha=0.2, color='gray', label='Day Range (High-Low)')
                ax1.set_title('NIFTY 50 Price Movement', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Price (₹)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper left')
                if plot_type == "Daily (All Data)":
                    if len(df_plot) >= 20:
                        df_plot['MA20'] = df_plot['close'].rolling(window=20).mean()
                        ax1.plot(df_plot['date'], df_plot['MA20'], label='20-Day MA', color='red', linestyle='--', alpha=0.8)
                    if len(df_plot) >= 50:
                        df_plot['MA50'] = df_plot['close'].rolling(window=50).mean()
                        ax1.plot(df_plot['date'], df_plot['MA50'], label='50-Day MA', color='green', linestyle='--', alpha=0.8)
                    ax1.legend(loc='upper left')
                if 'volume' in df_plot.columns and df_plot['volume'].notna().any() and df_plot['volume'].sum() > 0:
                    ax2.bar(df_plot['date'], df_plot['volume'], alpha=0.6, color='purple')
                    ax2.set_title(f'Trading Volume ({plot_type})', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Volume', fontsize=12)
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                else:
                    df_plot['daily_return'] = df_plot['close'].pct_change() * 100
                    ax2.plot(df_plot['date'], df_plot['daily_return'], color='red', alpha=0.7, linewidth=1)
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Return (%)', fontsize=12)
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.header("📊 Period Statistics")
                price_change = df['close'].iloc[-1] - df['close'].iloc[0]
                price_change_pct = (price_change / df['close'].iloc[0]) * 100
                volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Period Return", f"{price_change_pct:.2f}%", delta=f"₹{price_change:.2f}", help="Total return for the selected period")
                with col2:
                    st.metric("Highest Price", f"₹{df['high'].max():.2f}", help="Highest price during the period")
                with col3:
                    st.metric("Lowest Price", f"₹{df['low'].min():.2f}", help="Lowest price during the period")
                with col4:
                    st.metric("Volatility (Annual)", f"{volatility:.1f}%", help="Annualized volatility based on daily returns")
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
                with st.expander("📋 View Raw Data (Original Dataset)"):
                    st.write(f"Showing sample of {min(1000, len(df))} rows from {len(df):,} total records")
                    display_df = df.head(1000) if len(df) > 1000 else df
                    st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)
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
    st.markdown("---")
    st.markdown(
        "**NIFTY 50 Dashboard** | Built with Streamlit & Supabase | Data updated every 10 minutes",
        help="This dashboard provides real-time analysis of NIFTY 50 index data"
    )
