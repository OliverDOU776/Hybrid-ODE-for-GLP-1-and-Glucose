"""
Download and preprocess MIMIC-IV glucose and insulin data from PhysioNet.

This script requires valid PhysioNet credentials to access MIMIC-IV data.
It downloads relevant tables, interpolates to 5-minute intervals, and saves
the processed data as a Parquet file.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import argparse
import logging
import getpass
import requests
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MIMICDownloader:
    """
    Download and preprocess MIMIC-IV data for glucose-insulin modeling.
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None,
                 output_dir: str = "./data"):
        """
        Initialize MIMIC downloader.
        
        Args:
            username: PhysioNet username
            password: PhysioNet password
            output_dir: Directory to save processed data
        """
        self.username = username
        self.password = password
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # MIMIC-IV data paths (placeholder - actual implementation would use PhysioNet API)
        self.base_url = "https://physionet.org/files/mimiciv/2.2/"
        
    def check_credentials(self) -> bool:
        """
        Check if PhysioNet credentials are valid.
        
        Returns:
            valid: Whether credentials are valid
        """
        if not self.username or not self.password:
            logger.error("PhysioNet credentials not provided")
            return False
            
        # Placeholder check - actual implementation would verify with PhysioNet
        logger.info("Checking PhysioNet credentials...")
        logger.warning("Credential verification not implemented - using placeholder")
        return True
    
    def download_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Download relevant MIMIC-IV tables.
        
        Returns:
            tables: Dictionary of downloaded tables
        """
        logger.info("Downloading MIMIC-IV tables...")
        
        # Tables we need:
        # - labevents: For glucose and insulin measurements
        # - d_labitems: For lab item definitions
        # - patients: For patient demographics
        # - admissions: For admission information
        
        tables = {}
        
        # Placeholder - actual implementation would download from PhysioNet
        logger.warning("MIMIC download not implemented - creating synthetic data instead")
        
        # Create synthetic MIMIC-like data for demonstration
        n_patients = 100
        n_measurements_per_patient = 50
        
        # Generate synthetic glucose and insulin data
        data_records = []
        
        for patient_id in range(n_patients):
            # Generate time series for this patient
            base_time = datetime(2020, 1, 1) + timedelta(days=patient_id)
            
            # Patient characteristics
            has_diabetes = np.random.random() < 0.3  # 30% diabetes prevalence
            base_glucose = 7.0 if has_diabetes else 5.5  # mmol/L
            base_insulin = 80.0 if has_diabetes else 60.0  # pmol/L
            
            for i in range(n_measurements_per_patient):
                time = base_time + timedelta(minutes=5 * i)
                
                # Simulate meal effects
                hour = time.hour
                meal_effect = 0
                if 7 <= hour <= 8:  # Breakfast
                    meal_effect = 2.0 * np.exp(-0.1 * (i - 10))
                elif 12 <= hour <= 13:  # Lunch
                    meal_effect = 1.5 * np.exp(-0.1 * (i - 30))
                elif 18 <= hour <= 19:  # Dinner
                    meal_effect = 1.8 * np.exp(-0.1 * (i - 50))
                
                # Add variability and meal effects
                glucose = base_glucose + meal_effect + np.random.normal(0, 0.5)
                insulin = base_insulin + meal_effect * 20 + np.random.normal(0, 10)
                
                # Add GLP-1 and glucagon (synthetic)
                glp1 = 10 + meal_effect * 5 + np.random.normal(0, 2)
                glucagon = 80 - meal_effect * 10 + np.random.normal(0, 5)
                
                record = {
                    'subject_id': patient_id,
                    'hadm_id': patient_id * 1000,
                    'charttime': time,
                    'glucose_mmol_L': max(2.0, glucose),
                    'insulin_pmol_L': max(0.0, insulin),
                    'glp1_pmol_L': max(0.0, glp1),
                    'glucagon_pmol_L': max(0.0, glucagon),
                    'has_diabetes': has_diabetes
                }
                data_records.append(record)
        
        tables['measurements'] = pd.DataFrame(data_records)
        
        return tables
    
    def interpolate_to_grid(self, df: pd.DataFrame, 
                           interval_minutes: int = 5) -> pd.DataFrame:
        """
        Interpolate measurements to regular time grid.
        
        Args:
            df: DataFrame with measurements
            interval_minutes: Time interval in minutes
            
        Returns:
            interpolated: DataFrame with regular time grid
        """
        logger.info(f"Interpolating to {interval_minutes}-minute grid...")
        
        interpolated_dfs = []
        
        for subject_id in df['subject_id'].unique():
            subject_data = df[df['subject_id'] == subject_id].copy()
            subject_data = subject_data.sort_values('charttime')
            
            # Set time as index
            subject_data.set_index('charttime', inplace=True)
            
            # Create regular time grid
            start_time = subject_data.index.min()
            end_time = subject_data.index.max()
            time_grid = pd.date_range(start=start_time, end=end_time, 
                                    freq=f'{interval_minutes}min')
            
            # Interpolate numeric columns
            numeric_cols = ['glucose_mmol_L', 'insulin_pmol_L', 'glp1_pmol_L', 'glucagon_pmol_L']
            
            # Reindex to time grid and interpolate
            subject_data = subject_data.reindex(time_grid)
            subject_data[numeric_cols] = subject_data[numeric_cols].interpolate(method='linear')
            
            # Forward fill non-numeric columns
            non_numeric_cols = [col for col in subject_data.columns if col not in numeric_cols]
            subject_data[non_numeric_cols] = subject_data[non_numeric_cols].fillna(method='ffill')
            
            # Add back subject_id
            subject_data['subject_id'] = subject_id
            
            # Reset index
            subject_data.reset_index(inplace=True)
            subject_data.rename(columns={'index': 'charttime'}, inplace=True)
            
            interpolated_dfs.append(subject_data)
        
        interpolated = pd.concat(interpolated_dfs, ignore_index=True)
        
        return interpolated
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = "mimic_glucose_insulin.parquet"):
        """
        Save processed data to Parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving to {output_path}...")
        
        # Add metadata
        metadata = {
            'description': 'MIMIC-IV glucose and insulin data (synthetic placeholder)',
            'interval_minutes': '5',
            'created_date': datetime.now().isoformat(),
            'n_subjects': str(df['subject_id'].nunique()),
            'n_records': str(len(df))
        }
        
        # Convert to PyArrow table with metadata
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata(metadata)
        
        # Write to Parquet
        pq.write_table(table, output_path, compression='snappy')
        
        logger.info(f"Data saved successfully to {output_path}")
        logger.info(f"  - Subjects: {df['subject_id'].nunique()}")
        logger.info(f"  - Records: {len(df)}")
        logger.info(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def run(self):
        """
        Run the complete download and preprocessing pipeline.
        """
        logger.info("Starting MIMIC-IV data download and preprocessing...")
        
        # Check credentials
        if not self.check_credentials():
            logger.error("Invalid credentials. Please set PhysioNet username and password.")
            logger.info("You can sign up at: https://physionet.org/register/")
            logger.info("After registration, complete required training at: https://physionet.org/content/mimiciv/")
            return
        
        # Download tables
        tables = self.download_tables()
        
        # Process measurements
        measurements = tables['measurements']
        
        # Interpolate to regular grid
        interpolated = self.interpolate_to_grid(measurements)
        
        # Save to Parquet
        self.save_to_parquet(interpolated)
        
        logger.info("Processing complete!")


def main():
    """
    Main entry point for CLI.
    """
    parser = argparse.ArgumentParser(
        description="Download and preprocess MIMIC-IV glucose/insulin data"
    )
    parser.add_argument('--username', type=str, help='PhysioNet username')
    parser.add_argument('--password', type=str, help='PhysioNet password')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory (default: ./data)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Interpolation interval in minutes (default: 5)')
    
    args = parser.parse_args()
    
    # Get credentials if not provided
    username = args.username
    password = args.password
    
    if not username:
        username = input("PhysioNet username: ")
    if not password:
        password = getpass.getpass("PhysioNet password: ")
    
    # Create downloader and run
    downloader = MIMICDownloader(username=username, password=password,
                               output_dir=args.output_dir)
    downloader.run()


if __name__ == "__main__":
    main()