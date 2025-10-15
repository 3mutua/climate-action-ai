import pandas as pd
import numpy as np
import requests
import os
from typing import Optional, Dict, Any

class ClimateDataLoader:
    """
    Data loader for climate and economic indicators
    Supports both local CSV files and remote data sources
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # World Bank API endpoints for real data
        self.world_bank_indicators = {
            'gdp_per_capita': 'NY.GDP.PCAP.CD',
            'population': 'SP.POP.TOTL',
            'energy_use': 'EG.USE.PCAP.KG.OE',
            'co2_emissions': 'EN.ATM.CO2E.KT',
            'renewable_energy': 'EG.FEC.RNEW.ZS',
            'urban_population': 'SP.URB.TOTL.IN.ZS'
        }
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load the sample climate data from CSV
        """
        filepath = os.path.join(self.data_dir, 'sample_climate_data.csv')
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"✅ Loaded sample data with {len(df)} records")
            return df
        else:
            print("❌ Sample data file not found. Generating sample data...")
            return self.generate_sample_data()
    
    def generate_sample_data(self, n_samples: int = 200) -> pd.DataFrame:
        """
        Generate synthetic climate data for demonstration
        """
        np.random.seed(42)
        
        data = {
            'country': [f'Country_{i}' for i in range(n_samples)],
            'gdp_per_capita': np.random.normal(15000, 8000, n_samples),
            'population': np.random.normal(50, 30, n_samples),
            'energy_consumption': np.random.normal(80, 40, n_samples),
            'renewable_ratio': np.random.uniform(0.05, 0.6, n_samples),
            'industrial_output': np.random.normal(30, 15, n_samples),
            'urbanization_rate': np.random.uniform(0.3, 0.9, n_samples),
        }
        
        # Generate realistic carbon emissions based on features
        data['carbon_emissions'] = (
            0.4 * data['gdp_per_capita'] / 1000 +
            0.3 * data['energy_consumption'] * (1 - data['renewable_ratio']) +
            0.2 * data['industrial_output'] +
            0.1 * data['population'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Ensure emissions are positive
        data['carbon_emissions'] = np.abs(data['carbon_emissions'])
        
        df = pd.DataFrame(data)
        
        # Save generated data
        filepath = os.path.join(self.data_dir, 'sample_climate_data.csv')
        df.to_csv(filepath, index=False)
        print(f"✅ Generated and saved sample data with {len(df)} records")
        
        return df
    
    def load_world_bank_data(self, country_codes: list = ['USA', 'CHN', 'IND', 'BRA', 'ZAF']) -> pd.DataFrame:
        """
        Load real data from World Bank API
        Note: This requires internet connection
        """
        print("🌐 Fetching data from World Bank API...")
        
        all_data = []
        
        for country_code in country_codes:
            country_data = self._fetch_country_data(country_code)
            if country_data:
                all_data.append(country_data)
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"✅ Loaded real data for {len(df)} countries")
            return df
        else:
            print("❌ Failed to fetch World Bank data. Using sample data instead.")
            return self.load_sample_data()
    
    def _fetch_country_data(self, country_code: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a single country from World Bank API
        """
        try:
            base_url = "http://api.worldbank.org/v2/country"
            
            # Fetch country metadata
            metadata_url = f"{base_url}/{country_code}?format=json"
            metadata_response = requests.get(metadata_url, timeout=10)
            
            if metadata_response.status_code != 200:
                return None
            
            metadata = metadata_response.json()[1][0] if len(metadata_response.json()) > 1 else None
            if not metadata:
                return None
            
            country_data = {
                'country': metadata['name'],
                'country_code': country_code,
                'region': metadata.get('region', {}).get('value', 'Unknown'),
                'income_level': metadata.get('incomeLevel', {}).get('value', 'Unknown')
            }
            
            # Fetch indicator data (latest available)
            for indicator_name, indicator_code in self.world_bank_indicators.items():
                indicator_url = f"{base_url}/{country_code}/indicator/{indicator_code}?format=json&date=2020:2022"
                response = requests.get(indicator_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        # Get the most recent non-null value
                        values = [item['value'] for item in data[1] if item['value'] is not None]
                        if values:
                            country_data[indicator_name] = values[0]
                        else:
                            country_data[indicator_name] = None
                    else:
                        country_data[indicator_name] = None
                else:
                    country_data[indicator_name] = None
            
            # Calculate derived features
            if country_data.get('urban_population') and country_data.get('population'):
                country_data['urbanization_rate'] = country_data['urban_population'] / 100
            else:
                country_data['urbanization_rate'] = np.random.uniform(0.3, 0.9)
            
            if country_data.get('renewable_energy'):
                country_data['renewable_ratio'] = country_data['renewable_energy'] / 100
            else:
                country_data['renewable_ratio'] = np.random.uniform(0.05, 0.6)
            
            # Estimate industrial output (not directly available)
            country_data['industrial_output'] = np.random.normal(25, 10)
            
            return country_data
            
        except Exception as e:
            print(f"⚠️ Error fetching data for {country_code}: {e}")
            return None
    
    def get_country_profiles(self) -> pd.DataFrame:
        """
        Get predefined country profiles for demonstration
        """
        profiles = {
            'Developed_HighTech': {
                'gdp_per_capita': 45000,
                'population': 35,
                'energy_consumption': 120,
                'renewable_ratio': 0.35,
                'industrial_output': 25,
                'urbanization_rate': 0.85
            },
            'Developing_Industrial': {
                'gdp_per_capita': 12000,
                'population': 65,
                'energy_consumption': 85,
                'renewable_ratio': 0.15,
                'industrial_output': 35,
                'urbanization_rate': 0.55
            },
            'Emerging_Green': {
                'gdp_per_capita': 18000,
                'population': 45,
                'energy_consumption': 75,
                'renewable_ratio': 0.45,
                'industrial_output': 28,
                'urbanization_rate': 0.70
            },
            'LeastDeveloped_Agrarian': {
                'gdp_per_capita': 2800,
                'population': 25,
                'energy_consumption': 35,
                'renewable_ratio': 0.08,
                'industrial_output': 18,
                'urbanization_rate': 0.35
            }
        }
        
        return pd.DataFrame.from_dict(profiles, orient='index')
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded dataset for required columns and data quality
        """
        required_columns = [
            'gdp_per_capita', 'population', 'energy_consumption',
            'renewable_ratio', 'industrial_output', 'urbanization_rate',
            'carbon_emissions'
        ]
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            print(f"⚠️ Found null values: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data ranges
        validation_rules = {
            'gdp_per_capita': (500, 100000),
            'population': (0.1, 500),
            'energy_consumption': (1, 300),
            'renewable_ratio': (0, 1),
            'industrial_output': (5, 60),
            'urbanization_rate': (0.1, 1),
            'carbon_emissions': (0.1, 100)
        }
        
        for column, (min_val, max_val) in validation_rules.items():
            if column in df.columns:
                out_of_range = ~df[column].between(min_val, max_val)
                if out_of_range.any():
                    print(f"⚠️ Values out of range in {column}: {out_of_range.sum()} records")
        
        print("✅ Data validation completed")
        return True

# Utility functions
def explore_dataset(df: pd.DataFrame) -> None:
    """
    Generate basic exploration of the dataset
    """
    print("📊 Dataset Exploration")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlation with target
    if 'carbon_emissions' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['carbon_emissions'].sort_values(ascending=False)
        print("\nCorrelation with Carbon Emissions:")
        for feature, corr in correlations.items():
            if feature != 'carbon_emissions':
                print(f"  {feature}: {corr:.3f}")

def save_processed_data(df: pd.DataFrame, filename: str = "processed_climate_data.csv") -> None:
    """
    Save processed dataset with additional features
    """
    # Add derived features
    df_processed = df.copy()
    
    if 'gdp_per_capita' in df.columns and 'energy_consumption' in df.columns:
        df_processed['energy_intensity'] = df['energy_consumption'] / df['gdp_per_capita']
    
    if 'carbon_emissions' in df.columns and 'population' in df.columns:
        df_processed['emissions_per_capita'] = df['carbon_emissions'] / df['population']
    
    if 'renewable_ratio' in df.columns and 'energy_consumption' in df.columns:
        df_processed['fossil_energy'] = df['energy_consumption'] * (1 - df['renewable_ratio'])
    
    filepath = os.path.join("data", filename)
    df_processed.to_csv(filepath, index=False)
    print(f"✅ Processed data saved to {filepath}")
    
    return df_processed

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = ClimateDataLoader()
    
    # Load sample data
    df = loader.load_sample_data()
    
    # Explore the dataset
    explore_dataset(df)
    
    # Validate data
    loader.validate_data(df)
    
    # Save processed version
    processed_df = save_processed_data(df)
    
    print("\n🎉 Data loading demo completed!")
    print(f"Sample data shape: {df.shape}")
    print(f"Available countries: {df['country'].numpy()[:5]}...")  # Show first 5 countries