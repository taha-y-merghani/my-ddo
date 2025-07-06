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
6. Map Visualization

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
import folium
from folium import plugins
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

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
        
        # Common disease keywords (expanded from original analysis)
        disease_keywords = [
            'Zika', 'Hepatitis', 'Mumps', 'Measles', 'Influenza', 'Flu',
            'Cholera', 'Dengue', 'Malaria', 'Ebola', 'COVID', 'MERS',
            'SARS', 'H1N1', 'H5N1', 'Tuberculosis', 'TB', 'Mad Cow',
            'West Nile', 'Yellow Fever'
        ]
        
        # Extract diseases (case-insensitive)
        diseases = []
        text_lower = text.lower()
        for disease in disease_keywords:
            if disease.lower() in text_lower:
                diseases.append(disease)
        
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
    
    # Drop rows without coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"\n✓ Successfully geocoded {len(df)} locations")
    
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
    
    # Create enhanced visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Daily and Weekly Reports
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    daily_reports.plot(ax=ax1, alpha=0.5, label='Daily', color='lightgray')
    weekly_reports.plot(ax=ax1, linewidth=2, label='Weekly', color='blue')
    ax1.set_title('Disease Outbreak Reports Over Time', fontsize=14, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Number of Reports', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Disease Distribution Over Time (Stacked Area)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    disease_weekly.plot(ax=ax2, kind='area', stacked=True, alpha=0.7)
    ax2.set_title('Disease Distribution Over Time', fontsize=14, pad=20)
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Number of Reports', fontsize=12)
    ax2.legend(title='Diseases', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Geographic Distribution (Top 10 Locations)
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    location_counts = df['location'].value_counts().head(10)
    location_counts.plot(kind='bar', ax=ax3, color='skyblue')
    ax3.set_title('Top 10 Locations', fontsize=14, pad=20)
    ax3.set_xlabel('Location', fontsize=12)
    ax3.set_ylabel('Number of Reports', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Disease Distribution (Pie Chart)
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    disease_counts = df['disease'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    disease_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', colors=colors)
    ax4.set_title('Disease Distribution', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('outbreak_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Enhanced visualizations saved as 'outbreak_analysis.png'")
    
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

def cluster_outbreaks(df):
    """Cluster outbreak locations using DBSCAN"""
    print_section("4. Spatial Clustering")
    
    # Prepare coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Custom distance metric using geodesic distance (in km)
    def gc_dist(a, b):
        return geodesic((a[0], a[1]), (b[0], b[1])).kilometers
    
    # Apply DBSCAN clustering (eps ≈ 400km, min_samples=3)
    clustering = DBSCAN(eps=4, min_samples=3, metric=gc_dist)
    df['cluster'] = clustering.fit_predict(coords)
    
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    print(f"✓ Found {n_clusters} clusters")
    print(f"✓ {(df['cluster'] == -1).sum()} points classified as noise")
    
    return df

def create_disease_map(df):
    """Create an interactive map visualization of disease outbreaks with clusters"""
    print_section("5. Map Visualization")
    
    # Create color maps
    diseases = df['disease'].unique()
    cluster_ids = sorted(set(df['cluster']))
    
    # Color schemes
    disease_colors = plt.cm.Set3(np.linspace(0, 1, len(diseases)))
    disease_color_map = dict(zip(diseases, [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                                          for r, g, b, _ in disease_colors]))
    
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_ids)))
    cluster_color_map = dict(zip(cluster_ids, [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                                             for r, g, b, _ in cluster_colors]))
    
    # Create base map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Add disease markers with cluster information
    for idx, row in df.iterrows():
        disease_color = disease_color_map.get(row['disease'], '#808080')
        cluster_color = cluster_color_map.get(row['cluster'], '#808080')
        
        # Create marker with both disease and cluster information
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"""
                <b>Location:</b> {row['location']}<br>
                <b>Disease:</b> {row['disease']}<br>
                <b>Cluster:</b> {row['cluster']}<br>
                <b>Date:</b> {row['date'].strftime('%Y-%m-%d')}
            """,
            color=disease_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legends
    legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
        padding: 10px; border: 2px solid grey; border-radius: 5px">
        <h4>Disease Types</h4>
    '''
    for disease, color in disease_color_map.items():
        if pd.notna(disease):  # Skip None/NaN
            legend_html += f'''
                <div>
                    <span style="background-color: {color}; display: inline-block; width: 12px; height: 12px; 
                    border-radius: 50%; margin-right: 5px;"></span>
                    {disease}
                </div>
            '''
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add cluster legend
    cluster_legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
        padding: 10px; border: 2px solid grey; border-radius: 5px">
        <h4>Clusters</h4>
    '''
    for cluster_id, color in cluster_color_map.items():
        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        cluster_legend_html += f'''
            <div>
                <span style="background-color: {color}; display: inline-block; width: 12px; height: 12px; 
                border-radius: 50%; margin-right: 5px;"></span>
                {label}
            </div>
        '''
    cluster_legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(cluster_legend_html))
    
    # Save map
    m.save('outbreak_map.html')
    print("✓ Interactive map saved as 'outbreak_map.html'")
    
    return m

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
        df = cluster_outbreaks(df)
        create_disease_map(df)
        
        print("\nAnalysis complete! 🎉")
        print("\nOpen 'outbreak_map.html' in your browser to view the interactive visualization.")
    else:
        print("Analysis aborted due to data loading error.")

if __name__ == "__main__":
    main() 