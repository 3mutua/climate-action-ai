import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.emission_predictor import ClimateEmissionPredictor
from data.load_data import ClimateDataLoader

def train_with_real_data():
    """Train model using data loader"""
    print("🚀 Starting model training with climate data...")
    
    # Load data
    loader = ClimateDataLoader()
    df = loader.load_sample_data()
    
    # Explore data
    print(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Initialize and train model
    predictor = ClimateEmissionPredictor(model_type='random_forest')
    results = predictor.train(df)
    
    print("\n🎯 Training Results:")
    print(f"R² Score: {results['r2']:.3f}")
    print(f"MAE: {results['mae']:.3f}")
    print(f"Train Score: {results['train_score']:.3f}")
    print(f"Test Score: {results['test_score']:.3f}")
    
    if results['feature_importance']:
        print("\n🔍 Feature Importance:")
        for feature, importance in results['feature_importance'].items():
            print(f"  {feature}: {importance:.3f}")
    
    # Save model
    predictor.save_model('models/emission_model.pkl')
    print("✅ Model saved as 'models/emission_model.pkl'")
    
    return predictor, results, df

def create_training_report(predictor, results, df, output_file='training_report.html'):
    """Generate a comprehensive training report"""
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature importance
    if results['feature_importance']:
        importance_df = pd.DataFrame.from_dict(
            results['feature_importance'], 
            orient='index', 
            columns=['importance']
        )
        importance_df.sort_values('importance', ascending=True).plot(
            kind='barh', ax=axes[0,0], legend=False
        )
        axes[0,0].set_title('Feature Importance')
        axes[0,0].set_xlabel('Importance Score')
    
    # 2. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[0,1])
    axes[0,1].set_title('Feature Correlation Matrix')
    
    # 3. Emissions distribution
    axes[1,0].hist(df['carbon_emissions'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].set_xlabel('Carbon Emissions (megatons)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Carbon Emissions')
    
    # 4. Renewable energy vs Emissions
    axes[1,1].scatter(df['renewable_ratio'], df['carbon_emissions'], alpha=0.6)
    axes[1,1].set_xlabel('Renewable Energy Ratio')
    axes[1,1].set_ylabel('Carbon Emissions')
    axes[1,1].set_title('Renewable Energy Impact on Emissions')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create text report
    report = f"""
    # Climate Action AI - Model Training Report
    ## UN SDG 13: Carbon Emission Prediction Model
    
    ### Model Performance
    - **Algorithm**: Random Forest Regressor
    - **R² Score**: {results['r2']:.3f}
    - **Mean Absolute Error**: {results['mae']:.3f} megatons
    - **Training Score**: {results['train_score']:.3f}
    - **Testing Score**: {results['test_score']:.3f}
    
    ### Dataset Summary
    - **Total Samples**: {len(df)}
    - **Features**: {len(predictor.feature_names)}
    - **Features Used**: {', '.join(predictor.feature_names)}
    
    ### Key Insights
    """
    
    if results['feature_importance']:
        report += "\n#### Feature Importance Rankings:\n"
        for feature, importance in results['feature_importance'].items():
            report += f"- **{feature}**: {importance:.1%}\n"
    
    # Save report
    with open('training_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Training report generated: 'training_report.md'")
    print("✅ Visualization saved: 'training_analysis.png'")

if __name__ == "__main__":
    # Train model with data
    predictor, results, df = train_with_real_data()
    
    # Generate comprehensive report
    create_training_report(predictor, results, df)
    
    # Test predictions
    print("\n🧪 Testing Predictions:")
    
    test_profiles = [
        {
            'gdp_per_capita': 15000,
            'population': 50,
            'energy_consumption': 80,
            'renewable_ratio': 0.2,
            'industrial_output': 30,
            'urbanization_rate': 0.6
        },
        {
            'gdp_per_capita': 8000,
            'population': 25,
            'energy_consumption': 45,
            'renewable_ratio': 0.4,
            'industrial_output': 22,
            'urbanization_rate': 0.45
        }
    ]
    
    for i, profile in enumerate(test_profiles):
        prediction = predictor.predict(profile)
        print(f"Test {i+1}: {prediction:.2f} megatons CO₂")
    
    print("\n🎉 Model training and evaluation completed!")