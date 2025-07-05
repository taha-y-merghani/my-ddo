"""
Generate synthetic headlines for disease outbreak analysis using LLM templates.
This script creates realistic-looking headlines about a Zika-like disease outbreak
that can be used for testing the DDO (Discovering Disease Outbreaks) pipeline.
"""

import random
from datetime import datetime, timedelta
import json

# Template components for headline generation
LOCATIONS = [
    "Miami, Florida", "San Juan, Puerto Rico", "Rio de Janeiro, Brazil",
    "Mexico City, Mexico", "Singapore", "Bangkok, Thailand",
    "Manila, Philippines", "Jakarta, Indonesia"
]

TEMPLATES = [
    "New {disease} cases reported in {location}: {number} infected",
    "Health officials confirm {disease} outbreak in {location}",
    "{location} reports spike in {disease} infections, {number} new cases",
    "{disease} spreads in {location}, authorities on high alert",
    "Breaking: {number} {disease} cases identified in {location}",
]

DISEASE_VARIANTS = [
    "Zika", "Zika-like virus", "mosquito-borne illness",
    "tropical fever", "viral infection"
]

def generate_date_range(start_date_str="2024-01-01", num_days=90):
    """Generate a list of dates within a range."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    dates = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d")
             for x in range(num_days)]
    return dates

def generate_headline(date):
    """Generate a single synthetic headline."""
    template = random.choice(TEMPLATES)
    location = random.choice(LOCATIONS)
    disease = random.choice(DISEASE_VARIANTS)
    number = random.randint(5, 100)
    
    headline = template.format(
        location=location,
        disease=disease,
        number=number
    )
    
    return f"{date}\t{headline}"

def generate_headlines(num_headlines=200, output_file="headlines.txt"):
    """Generate multiple headlines and save to file."""
    dates = generate_date_range(num_days=90)
    
    # Generate more headlines for some dates to simulate outbreak clusters
    headlines = []
    for _ in range(num_headlines):
        # Weight recent dates more heavily
        date_weights = [1 + i/len(dates) for i in range(len(dates))]
        date = random.choices(dates, weights=date_weights, k=1)[0]
        headlines.append(generate_headline(date))
    
    # Sort by date
    headlines.sort()
    
    # Save to file
    with open(output_file, "w") as f:
        f.write("\n".join(headlines))
    
    print(f"Generated {num_headlines} headlines and saved to {output_file}")

if __name__ == "__main__":
    generate_headlines() 