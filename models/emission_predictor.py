import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class ClimateEmissionPredictor:
    """
    Machine Learning model for predicting carbon emissions
    based on economic and energy indicators for SDG 13
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'gdp_per_capita', 'population', 'energy_consumption',
            'renewable_ratio', 'industrial_output', 'urbanization_rate'
        ]
        self.is_trained = False
        
    def create_sample_data(self, n_samples=1000):
        """
        Generate synthetic climate data for demonstration
        In real scenario, this would be replaced with actual World Bank/UN data
        """
        np.random.seed(42)
        
        data = {
            'country': [f'Country_{i}' for i in range(n_samples)],
            'gdp_per_capita': np.random.normal(15000, 8000, n_samples),
            'population': np.random.normal(50, 30, n_samples),  # in millions
            'energy_consumption': np.random.normal(80, 40, n_samples),  # MTOE
            'renewable_ratio': np.random.uniform(0.05, 0.6, n_samples),
            'industrial_output': np.random.normal(30, 15, n_samples),  # % of GDP
            'urbanization_rate': np.random.uniform(0.3, 0.9, n_samples),
        }
        
        # Synthetic carbon emissions (target variable)
        # Realistic formula with some noise
        data['carbon_emissions'] = (
            0.4 * data['gdp_per_capita'] / 1000 +
            0.3 * data['energy_consumption'] * (1 - data['renewable_ratio']) +
            0.2 * data['industrial_output'] +
            0.1 * data['population'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Ensure emissions are positive
        data['carbon_emissions'] = np.abs(data['carbon_emissions'])
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Clean and prepare data for training"""
        # Select features and target
        X = df[self.feature_names]
        y = df['carbon_emissions']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        return X, y
    
    def train(self, df=None):
        """Train the emission prediction model"""
        if df is None:
            df = self.create_sample_data()
        
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=1000,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'r2': r2,
            'feature_importance': self.get_feature_importance(X.columns) if hasattr(self.model, 'feature_importances_') else None
        }
    
    def predict(self, input_data):
        """Predict carbon emissions for new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value
        
        # Select and scale features
        X = input_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        
        return prediction[0] if isinstance(input_data, dict) else prediction
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return None
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
    
    def analyze_policy_impact(self, base_data, policy_changes):
        """
        Analyze impact of policy changes on carbon emissions
        """
        base_emission = self.predict(base_data)
        
        scenario_results = []
        for policy_name, changes in policy_changes.items():
            scenario_data = base_data.copy()
            for feature, change in changes.items():
                if feature in scenario_data:
                    if isinstance(change, str) and change.endswith('%'):
                        # Percentage change
                        percentage = float(change.strip('%')) / 100
                        scenario_data[feature] = scenario_data[feature] * (1 + percentage)
                    else:
                        # Absolute change
                        scenario_data[feature] = scenario_data[feature] + change
            
            scenario_emission = self.predict(scenario_data)
            reduction = base_emission - scenario_emission
            reduction_pct = (reduction / base_emission) * 100
            
            scenario_results.append({
                'policy': policy_name,
                'new_emission': scenario_emission,
                'reduction': reduction,
                'reduction_pct': reduction_pct
            })
        
        return scenario_results

# Example usage
if __name__ == "__main__":
    # Initialize and train model
    predictor = ClimateEmissionPredictor()
    results = predictor.train()
    
    print("Model Training Results:")
    print(f"R² Score: {results['r2']:.3f}")
    print(f"MAE: {results['mae']:.3f}")
    
    if results['feature_importance']:
        print("\nFeature Importance:")
        for feature, importance in results['feature_importance'].items():
            print(f"  {feature}: {importance:.3f}")
    
    # Example prediction
    sample_country = {
        'gdp_per_capita': 12000,
        'population': 45,
        'energy_consumption': 75,
        'renewable_ratio': 0.15,
        'industrial_output': 28,
        'urbanization_rate': 0.6
    }
    
    prediction = predictor.predict(sample_country)
    print(f"\nPredicted Carbon Emissions: {prediction:.2f} megatons")
    
    # Save model
    predictor.save_model('models/emission_model.pkl')