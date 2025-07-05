"""
Generate synthetic headlines for disease outbreak analysis using LLM prompts.
This script creates highly realistic headlines about a Zika-like disease outbreak
by using LLM prompts to generate varied and natural-sounding news headlines.
"""

import random
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HeadlineGenerator:
    def __init__(self):
        """Initialize the headline generator with geographic and temporal parameters."""
        self.locations = {
            "North America": ["Miami, Florida", "Houston, Texas", "San Juan, Puerto Rico"],
            "South America": ["Rio de Janeiro, Brazil", "São Paulo, Brazil", "Bogotá, Colombia"],
            "Asia": ["Singapore", "Bangkok, Thailand", "Manila, Philippines"],
            "Pacific": ["Jakarta, Indonesia", "Port Moresby, Papua New Guinea"]
        }
        
        self.disease_context = {
            "primary": "Zika",
            "variants": ["Zika-like virus", "mosquito-borne illness", "tropical fever"],
            "symptoms": ["fever", "rash", "joint pain", "red eyes"],
            "vectors": ["mosquitoes", "Aedes mosquitoes", "disease vectors"]
        }

    def generate_llm_prompt(self, date: str, location: str) -> str:
        """Generate a prompt for the LLM to create a realistic headline."""
        return f"""Generate a single, realistic news headline about a Zika-like virus outbreak.
Date: {date}
Location: {location}

Requirements:
- Make it sound like a real news headline
- Include specific location
- Optionally include case numbers (5-100 range)
- Focus on outbreak reporting, health alerts, or case confirmations
- Do not mention death counts
- Keep it under 100 characters

Format: Return ONLY the headline text, nothing else."""

    def get_llm_completion(self, prompt: str) -> str:
        """Get completion from LLM API. Replace with your preferred LLM API."""
        # This is a placeholder - replace with actual LLM API call
        # For example, using OpenAI's API:
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        return response.choices[0].text.strip()
        """
        # For now, return a template-based headline
        templates = [
            "Health Alert: Zika-like Virus Spreads in {location}",
            "Breaking: {location} Reports 47 New Cases of Tropical Fever",
            "Mosquito-borne Illness Outbreak Confirmed in {location}",
            "Public Health Officials Monitor Zika Surge in {location}"
        ]
        return random.choice(templates).format(location=prompt.split("Location: ")[1].split("\n")[0])

    def generate_headline(self, date: str) -> str:
        """Generate a single headline using LLM."""
        # Select location with weighted distribution
        region = random.choice(list(self.locations.keys()))
        location = random.choice(self.locations[region])
        
        # Generate headline using LLM
        prompt = self.generate_llm_prompt(date, location)
        headline = self.get_llm_completion(prompt)
        
        return f"{date}\t{headline}"

    def generate_dataset(self, 
                        num_headlines: int = 200,
                        start_date: str = "2024-01-01",
                        num_days: int = 90,
                        output_file: str = "llm_headlines.txt") -> None:
        """Generate a complete dataset of headlines."""
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        dates = [(start + timedelta(days=x)).strftime("%Y-%m-%d")
                for x in range(num_days)]
        
        # Generate headlines with temporal clustering
        headlines = []
        for _ in range(num_headlines):
            # Weight recent dates more heavily
            date_weights = [1 + i/len(dates) for i in range(len(dates))]
            date = random.choices(dates, weights=date_weights, k=1)[0]
            headlines.append(self.generate_headline(date))
        
        # Sort by date
        headlines.sort()
        
        # Save to file
        with open(output_file, "w") as f:
            f.write("\n".join(headlines))
        
        print(f"Generated {len(headlines)} headlines and saved to {output_file}")

def main():
    """Main function to generate headlines dataset."""
    generator = HeadlineGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main() 