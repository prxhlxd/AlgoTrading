import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta

def show_paper_dashboard():
    print("Loading Paper Review Dashboard...")
    st.title("📈 Algorithmic Trading & Market Behavior: A Review")
    st.subheader("🔍 Overview")
    st.markdown("""
    **Paper Title:** *Analyzing the Impact of Algorithmic Trading on Stock Market Behavior*  
    **Authors:** Lawrence Damilare Oyeniyi, Chinonye Esther Ugochukwu, Noluthando Zamanjomane Mhlongo  
    **Journal:** World Journal of Advanced Engineering Technology and Sciences, 2024  
    **DOI:** [10.30574/wjaets.2024.11.2.0136](https://doi.org/10.30574/wjaets.2024.11.2.0136)

    This comprehensive review explores the evolution, strategies, impacts, and regulatory frameworks surrounding **algorithmic trading (AT)**.
    """)

    st.subheader("📌 Key Themes")
    st.markdown("""
    - **Evolution & Principles:** AT evolved with electronic trading and computational advances, automating decision-making and execution.
    - **Strategies:** Ranges from simple rules to complex machine learning models, including high-frequency trading (HFT).
    - **Market Liquidity:** Mixed outcomes — can both enhance and reduce liquidity depending on timeframes and conditions.
    - **Volatility:** AT may both dampen and amplify volatility; effects are context-specific.
    - **Regulation:** Needs to catch up with tech advancements. Emphasis on transparency, risk controls, and human oversight.
    - **Technology:** Big Data, AI, and machine learning are revolutionizing trading performance and strategy adaptability.
    """)

    st.subheader("📊 Empirical Insights")
    st.markdown("""
    - **Liquidity:** AT generally narrows spreads and enhances liquidity, especially in stable markets.
    - **Volatility:** Flash crashes (e.g., 2010) highlight the risk of instability from rapid algorithmic reactions.
    - **Trading Volume:** AT has increased volume by enabling faster, high-frequency executions.
    - **Performance Comparison:** AT outperforms traditional trading in speed, data utilization, and execution accuracy — but depends on context.

    > ⚠️ Excessive reliance can lead to market manipulation or crowding out of small investors.
    """)

    st.subheader("🧠 Research Gaps Identified")
    st.markdown("""
    - Lack of literature on **long-term socio-economic impacts**.
    - Insufficient studies on **human-algorithm interaction** in live markets.
    - Need for a **unified evaluation framework** for AT strategies.
    """)

    st.subheader("🛡️ Strategic Recommendations")
    st.markdown("""
    ### For Market Participants
    - Leverage **adaptive learning** and **big data** to optimize strategies.
    - Ensure **ethical compliance** and **system resilience**.

    ### For Regulators
    - Implement **circuit breakers**, stress testing, and **AI-aware oversight**.
    - Foster **collaborative regulation** among stakeholders and tech experts.

    > The study urges continuous dialogue between technologists, regulators, and investors to balance innovation and stability.
    """)

    st.subheader("📚 Methodology")
    st.markdown("""
    - **Approach:** Thematic analysis of peer-reviewed articles and case studies.
    - **Sources:** Empirical research on market behavior, simulation models, regulatory frameworks, and AI strategies.
    """)

    st.subheader("🧾 Conclusion")
    st.markdown("""
    Algorithmic trading is both a **technological breakthrough** and a **regulatory challenge**.  
    While it enhances **efficiency, liquidity, and strategy execution**, it also poses risks related to **volatility, inequality, and oversight**.

    > Embracing innovation responsibly is key to shaping the future of global financial markets.
    """)

# Example of usage
# show_paper_dashboard()
