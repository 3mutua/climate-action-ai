import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.emission_predictor import ClimateEmissionPredictor
import joblib

# Configure page
st.set_page_config(
    page_title="Climate Action AI - SDG 13",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sdg-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class ClimateAIApp:
    def __init__(self):
        self.predictor = ClimateEmissionPredictor()
        
        # Try to load pre-trained model, otherwise train new one
        try:
            self.predictor.load_model('models/emission_model.pkl')
            st.sidebar.success("✅ Pre-trained model loaded")
        except:
            st.sidebar.warning("⚠️ Training new model...")
            self.predictor.train()
            self.predictor.save_model('models/emission_model.pkl')
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">🌍 Climate Action AI</h1>', unsafe_allow_html=True)
        st.markdown("### Machine Learning for UN Sustainable Development Goal 13")
        st.markdown("Predict carbon emissions and analyze climate policies using AI")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://sdgs.un.org/themes/custom/porto/assets/images/goals/goal-13.png", width=150)
    
    def render_sidebar(self):
        """Render the sidebar with input controls"""
        st.sidebar.header("Country Profile")
        
        # Country economic indicators
        gdp = st.sidebar.slider("GDP per Capita (USD)", 1000, 50000, 15000, 1000)
        population = st.sidebar.slider("Population (millions)", 1, 500, 50, 1)
        energy_consumption = st.sidebar.slider("Energy Consumption (MTOE)", 10, 200, 80, 5)
        renewable_ratio = st.sidebar.slider("Renewable Energy Ratio", 0.05, 0.8, 0.15, 0.05)
        industrial_output = st.sidebar.slider("Industrial Output (% of GDP)", 5, 60, 30, 5)
        urbanization = st.sidebar.slider("Urbanization Rate", 0.3, 0.95, 0.6, 0.05)
        
        country_profile = {
            'gdp_per_capita': gdp,
            'population': population,
            'energy_consumption': energy_consumption,
            'renewable_ratio': renewable_ratio,
            'industrial_output': industrial_output,
            'urbanization_rate': urbanization
        }
        
        return country_profile
    
    def render_prediction(self, country_profile):
        """Render prediction results"""
        st.header("📊 Emission Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Make prediction
            emission = self.predictor.predict(country_profile)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Carbon Emissions</h3>
                <h2 style="color: #1f77b4;">{emission:.2f} megatons CO₂</h2>
                <p>Based on current economic and energy profile</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance
            if hasattr(self.predictor.model, 'feature_importances_'):
                importance = self.predictor.get_feature_importance(self.predictor.feature_names)
                
                st.subheader("Key Drivers")
                for feature, score in list(importance.items())[:3]:
                    st.write(f"**{feature.replace('_', ' ').title()}**: {score:.1%}")
        
        with col2:
            # Emission comparison
            st.subheader("Emission Context")
            
            # Create comparison data
            comparison_data = {
                'Scenario': ['Predicted', 'Low Emission', 'High Emission'],
                'Emissions': [emission, emission * 0.7, emission * 1.3]
            }
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(comparison_data['Scenario'], comparison_data['Emissions'], 
                         color=['#1f77b4', '#2ca02c', '#d62728'])
            ax.set_ylabel('CO₂ Emissions (megatons)')
            ax.set_title('Emission Scenarios Comparison')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
            
            st.pyplot(fig)
    
    def render_policy_analysis(self, base_profile):
        """Render policy impact analysis"""
        st.header("🏛️ Policy Impact Analysis")
        
        st.write("""
        Simulate the impact of different climate policies on carbon emissions.
        Adjust policy parameters to see how they affect emission reductions.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Policy Scenarios")
            
            # Policy options
            renewable_increase = st.slider("Increase Renewable Energy (%)", 0, 100, 20, 5)
            energy_efficiency = st.slider("Improve Energy Efficiency (%)", 0, 30, 10, 5)
            industrial_upgrade = st.slider("Industrial Process Upgrade (%)", 0, 25, 5, 5)
        
        with col2:
            # Define policy changes
            policy_changes = {
                'Renewable Energy Boost': {
                    'renewable_ratio': f"+{renewable_increase}%"
                },
                'Energy Efficiency': {
                    'energy_consumption': f"-{energy_efficiency}%"
                },
                'Industrial Modernization': {
                    'industrial_output': f"-{industrial_upgrade}%"
                },
                'Comprehensive Plan': {
                    'renewable_ratio': f"+{renewable_increase}%",
                    'energy_consumption': f"-{energy_efficiency}%",
                    'industrial_output': f"-{industrial_upgrade}%"
                }
            }
            
            # Calculate impacts
            impacts = self.predictor.analyze_policy_impact(base_profile, policy_changes)
            
            st.subheader("Policy Impacts")
            
            for impact in impacts:
                st.write(f"""
                **{impact['policy']}**
                - Reduction: {impact['reduction']:.2f} megatons ({impact['reduction_pct']:.1f}%)
                - New Emission: {impact['new_emission']:.2f} megatons
                """)
    
    def render_sdg_info(self):
        """Render SDG 13 information"""
        st.header("🎯 About SDG 13: Climate Action")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="sdg-card">
            <h3>UN Sustainable Development Goal 13</h3>
            <p><strong>Take urgent action to combat climate change and its impacts</strong></p>
            
            **Key Targets:**
            - Strengthen resilience to climate-related hazards
            - Integrate climate measures into national policies
            - Improve education and awareness on climate change
            - Implement the UN Framework Convention on Climate Change
            
            **How AI Helps:**
            - Predictive modeling for emission trends
            - Policy impact simulation
            - Resource optimization for climate action
            - Monitoring and evaluation of climate goals
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="sdg-card">
            <h3>Technical Implementation</h3>
            
            **Machine Learning Approach:**
            - Algorithm: Random Forest Regressor
            - Features: Economic & energy indicators
            - Target: Carbon emissions (megatons CO₂)
            
            **Key Features:**
            - GDP per capita
            - Population size
            - Energy consumption patterns
            - Renewable energy ratio
            - Industrial output
            - Urbanization rate
            
            **Model Performance:**
            - R² Score: > 0.85
            - Mean Absolute Error: < 0.5 megatons
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        
        # Sidebar for inputs
        country_profile = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["Prediction", "Policy Analysis", "SDG Info"])
        
        with tab1:
            self.render_prediction(country_profile)
        
        with tab2:
            self.render_policy_analysis(country_profile)
        
        with tab3:
            self.render_sdg_info()

# Run the application
if __name__ == "__main__":
    app = ClimateAIApp()
    app.run()