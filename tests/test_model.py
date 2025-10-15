import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.emission_predictor import ClimateEmissionPredictor
import pandas as pd
import numpy as np

class TestClimateEmissionPredictor(unittest.TestCase):
    
    def setUp(self):
        self.predictor = ClimateEmissionPredictor()
        self.sample_data = self.predictor.create_sample_data(100)
    
    def test_data_creation(self):
        """Test that sample data is created correctly"""
        self.assertEqual(len(self.sample_data), 100)
        self.assertIn('carbon_emissions', self.sample_data.columns)
        self.assertTrue(all(self.sample_data['carbon_emissions'] > 0))
    
    def test_model_training(self):
        """Test that model can be trained successfully"""
        results = self.predictor.train(self.sample_data)
        
        self.assertIn('r2', results)
        self.assertIn('mae', results)
        self.assertGreater(results['r2'], 0.5)  # Reasonable performance
        self.assertTrue(self.predictor.is_trained)
    
    def test_prediction(self):
        """Test prediction functionality"""
        # Train model first
        self.predictor.train(self.sample_data)
        
        # Test prediction with sample input
        test_input = {
            'gdp_per_capita': 10000,
            'population': 40,
            'energy_consumption': 60,
            'renewable_ratio': 0.3,
            'industrial_output': 25,
            'urbanization_rate': 0.5
        }
        
        prediction = self.predictor.predict(test_input)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)
    
    def test_policy_analysis(self):
        """Test policy impact analysis"""
        self.predictor.train(self.sample_data)
        
        base_profile = {
            'gdp_per_capita': 15000,
            'population': 50,
            'energy_consumption': 80,
            'renewable_ratio': 0.2,
            'industrial_output': 30,
            'urbanization_rate': 0.6
        }
        
        policy_changes = {
            'test_policy': {'renewable_ratio': '+10%'}
        }
        
        impacts = self.predictor.analyze_policy_impact(base_profile, policy_changes)
        
        self.assertEqual(len(impacts), 1)
        self.assertIn('reduction', impacts[0])
        self.assertIn('reduction_pct', impacts[0])

if __name__ == '__main__':
    unittest.main()