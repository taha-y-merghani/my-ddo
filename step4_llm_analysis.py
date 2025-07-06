#!/usr/bin/env python3
"""
Disease Outbreak Detection Analysis (LLM Data)
===========================================

This script replicates the analysis from the original step4 using LLM-generated headlines.
It follows these key steps:
1. Data Loading & Preprocessing
2. Entity Extraction (Locations & Diseases)
3. Geolocation
4. Time Series Analysis
5. Pattern Detection

The script outputs results to stdout for easy monitoring and debugging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import os

def print_section(title):
    """Helper function to print formatted section headers"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def load_data():
    """Load and preprocess the LLM-generated headlines"""
    print_section("1. Data Loading")
    
    try:
        with open('data/llm_headlines.txt', 'r') as f:
            headlines = f.readlines()
        
        # Clean headlines
        headlines = [h.strip() for h in headlines if h.strip()]
        
        # Create DataFrame with dates
        df = pd.DataFrame({
            'headline': headlines,
            'date': pd.date_range('2024-01-01', periods=len(headlines), freq='D')
        })
        
        print(f"✓ Successfully loaded {len(headlines)} headlines")
        print("\nSample headlines:")
        print(df.head().to_string())
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_entities(df, nlp):
    """Extract location and disease entities from headlines"""
    print_section("2. Entity Extraction")
    
    def process_headline(text):
        doc = nlp(text)
        
        # Extract locations (GPE = GeoPolitical Entity, LOC = Location)
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        
        # Common disease keywords
        disease_keywords = ['COVID', 'Flu', 'Influenza', 'Malaria', 'Dengue', 'Ebola',
                          'Zika', 'Cholera', 'Measles', 'Tuberculosis', 'TB',
                          'Pneumonia', 'HIV', 'AIDS', 'Hepatitis']
        
        # Extract diseases
        diseases = [word for word in text.split() 
                   if any(disease.lower() in word.lower() for disease in disease_keywords)]
        
        return {
            'location': locations[0] if locations else None,
            'disease': diseases[0] if diseases else None
        }
    
    print("Processing headlines with spaCy...")
    entities = [process_headline(headline) for headline in df['headline']]
    
    # Add extracted information to DataFrame
    df['location'] = [e['location'] for e in entities]
    df['disease'] = [e['disease'] for e in entities]
    
    print("\nExtraction Results:")
    print(f"✓ Found locations in {df['location'].notna().sum()} headlines")
    print(f"✓ Found diseases in {df['disease'].notna().sum()} headlines")
    print("\nSample processed headlines:")
    print(df[['headline', 'location', 'disease']].head().to_string())
    
    return df

def geocode_locations(df):
    """Geocode extracted locations"""
    print_section("3. Geolocation")
    
    geolocator = Nominatim(user_agent="disease_outbreak_detector")
    
    def geocode_location(location):
        if not location:
            return None
        
        try:
            location_data = geolocator.geocode(location, timeout=10)
            if location_data:
                return {
                    'latitude': location_data.latitude,
                    'longitude': location_data.longitude,
                    'address': location_data.address
                }
        except GeocoderTimedOut:
            print(f"⚠️  Geocoding timed out for {location}")
        except Exception as e:
            print(f"⚠️  Error geocoding {location}: {e}")
        
        return None
    
    # Geocode unique locations to avoid redundant API calls
    unique_locations = df['location'].dropna().unique()
    print(f"Processing {len(unique_locations)} unique locations...")
    
    location_cache = {}
    for loc in unique_locations:
        location_cache[loc] = geocode_location(loc)
        
    # Add geocoded information to DataFrame
    df['latitude'] = df['location'].map(lambda x: location_cache.get(x, {}).get('latitude'))
    df['longitude'] = df['location'].map(lambda x: location_cache.get(x, {}).get('longitude'))
    df['full_address'] = df['location'].map(lambda x: location_cache.get(x, {}).get('address'))
    
    print(f"\n✓ Successfully geocoded {df['latitude'].notna().sum()} locations")
    print("\nSample geocoded data:")
    print(df[['location', 'latitude', 'longitude']].head().to_string())
    
    return df

def analyze_time_series(df):
    """Perform time series analysis on the outbreak data"""
    print_section("4. Time Series Analysis")
    
    # Set date as index
    df_time = df.set_index('date').copy()
    
    # Calculate daily and weekly metrics
    daily_reports = df_time.resample('D').size()
    weekly_reports = df_time.resample('W').size()
    disease_weekly = df_time.groupby([pd.Grouper(freq='W'), 'disease']).size().unstack(fill_value=0)
    
    print("Time Series Summary:")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total reports: {len(df)}")
    print(f"Average daily reports: {daily_reports.mean():.1f}")
    print(f"Peak daily reports: {daily_reports.max()}")
    
    print("\nWeekly Disease Distribution:")
    print(disease_weekly.sum().sort_values(ascending=False))
    
    # Save time series plot
    plt.figure(figsize=(15, 8))
    daily_reports.plot(alpha=0.5, label='Daily')
    weekly_reports.plot(linewidth=2, label='Weekly')
    plt.title('Disease Outbreak Reports Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Reports')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('time_series_analysis.png')
    plt.close()
    
    print("\n✓ Time series plot saved as 'time_series_analysis.png'")
    
    return df_time

def analyze_patterns(df):
    """Identify patterns and trends in the outbreak data"""
    print_section("5. Pattern Analysis")
    
    # 1. Geographic Distribution
    location_counts = df['location'].value_counts()
    print("Top 10 Locations by Number of Reports:")
    print(location_counts.head(10))
    
    # 2. Disease Distribution
    disease_counts = df['disease'].value_counts()
    print("\nDisease Distribution:")
    print(disease_counts)
    
    # 3. Location-Disease Correlation
    print("\nTop Locations for Each Disease:")
    location_disease = pd.crosstab(df['location'], df['disease'])
    for disease in disease_counts.index:
        top_locations = location_disease[disease].nlargest(3)
        print(f"\n{disease}:")
        print(top_locations)

def main():
    """Main execution function"""
    print("\nStarting Disease Outbreak Detection Analysis...")
    print("Using LLM-generated headlines from data/llm_headlines.txt")
    
    # Load spaCy model
    print("\nLoading spaCy model...")
    try:
        nlp = spacy.load('en_core_web_sm')
        print("✓ spaCy model loaded successfully")
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        return
    
    # Execute analysis pipeline
    df = load_data()
    if df is not None:
        df = extract_entities(df, nlp)
        df = geocode_locations(df)
        df_time = analyze_time_series(df)
        analyze_patterns(df)
        
        print("\nAnalysis complete! 🎉")
    else:
        print("Analysis aborted due to data loading error.")

if __name__ == "__main__":
    main() 