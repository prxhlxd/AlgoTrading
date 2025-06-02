import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def show_paper_dashboard():
    # Page configuration
    # st.set_page_config(
    #     page_title="DLEF-SM Framework",
    #     page_icon="📈",
    #     layout="wide",
    #     initial_sidebar_state="expanded"
    # )

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .section-header {
            font-size: 2rem;
            color: #2c3e50;
            margin: 2rem 0 1rem 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .subsection-header {
            font-size: 1.5rem;
            color: #34495e;
            margin: 1.5rem 0 1rem 0;
        }
        .issue-box {
            background-color: #ED2939;
            border-left: 4px solid #e74c3c;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        .solution-box {
            background-color: #48AAAD;
            border-left: 4px solid #27ae60;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        .results-highlight {
            background-color: #48AAAD;
            border: 1px solid #ffeaa7;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .metric-card {
            background-color: #48AAAD;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    sections = ["Home", "Related Work", "Model Architecture", "Results & Discussion"]
    selected_section = st.sidebar.selectbox("Choose Section", sections)

    # Main title
    st.markdown('<h1 class="main-header">📈 DL Based Expert Framework For Portfolio Prediction</h1>', unsafe_allow_html=True)

    if selected_section == "Home":
        st.markdown("""
        ## Report on DLEF-SM
        This is a report on the paper titled "Deep Learning Based Expert Framework for Stock Market Prediction (DLEF-SM)".
        Link: [DLEF-SM Paper](https://www.researchgate.net/publication/382680950_A_Deep_Learning_Based_Expert_Framework_for_Portfolio_Prediction_and_Forecasting)
        
        ### Key Features:
        - **Advanced Preprocessing**: Improved Jellyfish-Induced Filtering (IJF-F)
        - **Feature Extraction**: ResNet-50 based feature extraction
        - **Optimization**: Improved Black Widow Optimization (IBWO)
        - **Prediction**: Deep Reinforcement Learning-Artificial Neural Network (DRL-ANN)
        
        Navigate through the sections using the sidebar to explore the complete framework.
        """)
    

    elif selected_section == "Related Work":
        st.markdown('<h2 class="section-header">Related Work and Their Drawbacks</h2>', unsafe_allow_html=True)
        
        # Issue 1
        st.markdown('<h3 class="subsection-header">Issue 1: High Computational Cost</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="issue-box">
            <strong>📊 Observed in:</strong> Lee et al. [23] used CNN models, which increased computation time and RAM usage because of the large input data and deep architecture.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="solution-box">
            <strong>💡 Proposed Solution:</strong> The <strong>Improved Jellyfish-Induced Filtering (IJF-F)</strong> in this paper performs good data preprocessing by removing high-frequency noise and irrelevant signals early in the workflow.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🎯 How it helps:** This reduces data dimensionality and complexity")
        
        # Issue 2
        st.markdown('<h3 class="subsection-header">Issue 2: Instability and Overfitting</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="issue-box">
            <strong>📊 Observed in:</strong> Ding et al. [22], where models showed unstable outputs due to sensitivity to small variations in training data.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="solution-box">
            <strong>💡 Proposed Solution:</strong> The <strong>Deep Reinforcement Learning-Artificial Neural Network (DRL-ANN)</strong> module in DLEF-SM dynamically adapts to new data through continuous learning.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🎯 How it helps:** DRL helps the model to learn optimal strategies using reward-based feedback, increasing generalization even in volatile finance data.")
        
        # Issue 3
        st.markdown('<h3 class="subsection-header">Issue 3: Ineffective Feature Selection</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="issue-box">
            <strong>📊 Observed in:</strong> Farahani et al. [29], algorithms lacked adaptability, leading to suboptimal feature sets.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="solution-box">
            <strong>💡 Proposed Solution:</strong> The <strong>Improved Black Widow Optimization (IBWO)</strong> algorithm in DLEF-SM uses adaptive mutation and diversity-aware selection to identify the most informative features.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🎯 How it helps:** This reduces underfitting and overfitting risks, ensures better model accuracy, and enhances prediction quality by dynamically selecting relevant features.")

    elif selected_section == "Model Architecture":
        st.markdown('<h2 class="section-header">Model Architecture/Framework</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        The framework follows a multiple stage workflow for stock market forecasting, combining preprocessing, 
        feature extraction, feature selection and deep learning.
        """)
        
        # Step-by-step process
        steps = [
            {
                "step": "1",
                "title": "Data Preprocessing",
                "description": "Raw time-series market data is processed through **Improved Jellyfish-Induced Filtering (IJF-F)** to remove noise and outliers by decomposing signals into low-frequency components.",
                "icon": "🔧"
            },
            {
                "step": "2", 
                "title": "Feature Extraction",
                "description": "This filtered data is given to a pre-trained **ResNet-50** model for **feature extraction**, treating time-series as 1D images to extract patterns. The resulting feature set is high-dimensional and may again increase computational cost.",
                "icon": "🎯"
            },
            {
                "step": "3",
                "title": "Feature Selection", 
                "description": "So **Improved Black Widow Optimization (IBWO)** selects the most relevant features.",
                "icon": "🔍"
            },
            {
                "step": "4",
                "title": "Deep Learning Prediction",
                "description": "The selected features are used to train a **Deep Reinforcement Learning--Artificial Neural Network (DRL-ANN)**, which learns to predict future stock movements by optimizing long-term reward through trial-and-error interactions.",
                "icon": "🧠"
            },
            {
                "step": "5",
                "title": "Evaluation",
                "description": "The system is evaluated by accuracy, RMSE, MAE, and Sharpe Ratio across datasets like SP500-S, SP500-L, and DAX.",
                "icon": "📊"
            }
        ]
        
        for step in steps:
            st.markdown(f"""
            ### {step['icon']} Step {step['step']}: {step['title']}
            {step['description']}
            """)
            st.markdown("---")

    elif selected_section == "Flowchart":
        st.markdown('<h2 class="section-header">Model Architecture Flowchart</h2>', unsafe_allow_html=True)
        
        # Create interactive flowchart using Plotly
        fig = go.Figure()
        
        # Define positions and connections
        nodes = [
            {"name": "Raw Time-Series\nMarket Data", "x": 1, "y": 5, "color": "#3498db"},
            {"name": "Improved Jellyfish-Induced\nFiltering (IJF-F)", "x": 3, "y": 5, "color": "#e74c3c"},
            {"name": "Filtered\nData", "x": 5, "y": 5, "color": "#95a5a6"},
            {"name": "ResNet-50\nFeature Extraction", "x": 7, "y": 5, "color": "#f39c12"},
            {"name": "High-Dimensional\nFeature Set", "x": 9, "y": 5, "color": "#95a5a6"},
            {"name": "Improved Black Widow\nOptimization (IBWO)", "x": 11, "y": 5, "color": "#9b59b6"},
            {"name": "Selected\nFeatures", "x": 13, "y": 5, "color": "#95a5a6"},
            {"name": "Deep Reinforcement Learning\nArtificial Neural Network\n(DRL-ANN)", "x": 15, "y": 5, "color": "#27ae60"},
            {"name": "Stock Market\nPredictions", "x": 17, "y": 5, "color": "#2c3e50"},
            {"name": "Evaluation Metrics:\n• Accuracy\n• RMSE\n• MAE\n• Sharpe Ratio", "x": 15, "y": 2, "color": "#34495e"}
        ]
        
        # Add nodes
        for node in nodes:
            fig.add_trace(go.Scatter(
                x=[node["x"]], 
                y=[node["y"]], 
                mode='markers+text',
                marker=dict(size=60, color=node["color"], line=dict(width=2, color='white')),
                text=node["name"],
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                showlegend=False,
                hoverinfo='text',
                hovertext=node["name"]
            ))
        
        # Add arrows between nodes
        arrows = [
            (1, 5, 3, 5), (3, 5, 5, 5), (5, 5, 7, 5), (7, 5, 9, 5),
            (9, 5, 11, 5), (11, 5, 13, 5), (13, 5, 15, 5), (15, 5, 17, 5),
            (15, 5, 15, 2)
        ]
        
        for arrow in arrows:
            fig.add_annotation(
                x=arrow[2], y=arrow[3],
                ax=arrow[0], ay=arrow[1],
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#2c3e50'
            )
        
        fig.update_layout(
            title="DLEF-SM Framework Architecture Flow",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 18]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1, 6]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            width=1200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add description below flowchart
        st.markdown("""
        ### Framework Components Description:
        
        1. **Raw Time-Series Market Data**: Input financial data containing historical stock prices and market indicators
        2. **IJF-F Preprocessing**: Noise reduction and signal decomposition for cleaner data
        3. **ResNet-50 Feature Extraction**: Deep learning-based pattern recognition treating time-series as 1D images
        4. **IBWO Feature Selection**: Optimization algorithm for selecting most relevant features
        5. **DRL-ANN Prediction**: Reinforcement learning model for stock movement prediction
        6. **Evaluation**: Performance assessment using multiple metrics across different datasets
        """)

    elif selected_section == "Results & Discussion":
        st.markdown('<h2 class="section-header">Results and Discussion</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        The results and discussion section of the paper evaluates the proposed DLEF-SM framework against 
        several traditional and deep learning models using:
        """)
        
        # Create tabs for different aspects of results
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Datasets & Models", "🔧 Preprocessing", "📈 Metrics", "🏆 Best Results"])
        
        with tab1:
            st.markdown("### Datasets Used:")
            datasets = ["SP500-S", "SP500-L", "DAX"]
            for i, dataset in enumerate(datasets, 1):
                st.markdown(f"{i}. **{dataset}**")
            
            st.markdown("### Comparison Models:")
            models = [
                "Random Forest (RF)", "Decision Tree (DT)", "Logistic Regression (LR)",
                "Support Vector Machine (SVM)", "Deep Neural Network (DNN)", 
                "Long Short-Term Memory (LSTM)", "Convolutional Neural Network (CNN)",
                "Double Q-Learning (DQN)", "Multi-DQN"
            ]
            
            col1, col2 = st.columns(2)
            for i, model in enumerate(models):
                if i < len(models)//2:
                    col1.markdown(f"• {model}")
                else:
                    col2.markdown(f"• {model}")
        
        with tab2:
            st.markdown("### Preprocessing Techniques Applied:")
            preprocessing_techniques = [
                "Wavelet Transform (WT)",
                "Singular Spectrum Analysis (SSA)", 
                "Kalman Filter"
            ]
            
            for i, technique in enumerate(preprocessing_techniques, 1):
                st.markdown(f"{i}. **{technique}**")
            
            st.info("Each preprocessing technique was applied in conjunction with each model for comprehensive comparison.")
        
        with tab3:
            st.markdown("### Evaluation Metrics:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Error Metrics:")
                st.markdown("• Mean Squared Error (MSE)")
                st.markdown("• Root Mean Squared Error (RMSE)")
                st.markdown("• Mean Absolute Error (MAE)")
            
            with col2:
                st.markdown("#### Quality Measures:")
                st.markdown("• Accuracy")
                st.markdown("• F1-score")
                st.markdown("• Precision")
                st.markdown("• Recall")
                st.markdown("• Minimum Drawdown (MDD)")
                st.markdown("• Sharpe Ratio (SR)")
        
        with tab4:
            st.markdown("### Outstanding Performance Results")
            
            st.markdown("""
            <div class="results-highlight">
            <strong>🏆 Best Configuration Achievement:</strong><br><br>
            The configuration using <strong>IJF-F with Wavelet Transform (WT) for preprocessing, VGGFace2 and ResNet-50 for feature extraction, IBWO for feature selection, and DRL-ANN for forecasting</strong> delivered the highest accuracy and lowest error rates.
            </div>
            """, unsafe_allow_html=True)
            
            # Create performance visualization
            datasets = ['SP500-S', 'SP500-L', 'DAX']
            accuracies = [99.562, 98.235, 98.825]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=datasets,
                y=accuracies,
                text=[f'{acc}%' for acc in accuracies],
                textposition='auto',
                marker_color=['#3498db', '#e74c3c', '#27ae60']
            ))
            
            fig.update_layout(
                title='DLEF-SM Framework Accuracy Across Datasets',
                xaxis_title='Datasets',
                yaxis_title='Accuracy (%)',
                yaxis=dict(range=[95, 100]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h2 style="color: #3498db;">99.562%</h2>
                    <p><strong>SP500-S Dataset</strong></p>
                    <p>Highest accuracy achieved</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h2 style="color: #e74c3c;">98.235%</h2>
                    <p><strong>SP500-L Dataset</strong></p>
                    <p>Long-term prediction accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h2 style="color: #27ae60;">98.825%</h2>
                    <p><strong>DAX Dataset</strong></p>
                    <p>European market performance</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("Across all datasets and preprocessing methods, the DLEF-SM framework consistently outperformed other models, making it the most effective setup for stock market prediction in the study.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
        <p>📈 DLEF-SM Framework - Deep Learning Based Expert Framework for Portfolio Prediction</p>
        <p>Advanced Stock Market Prediction using Deep Reinforcement Learning</p>
    </div>
    """, unsafe_allow_html=True)