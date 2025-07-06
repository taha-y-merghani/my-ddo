# Disease Discovery from Outbreaks (DDO) - A Comparative Analysis (2020 vs 2025)

## Project Overview
This repository is a revival and enhancement of the Manning liveProject "Discovering Disease Outbreaks from News Headlines". The project demonstrates the evolution of NLP and data science techniques in public health surveillance between 2020 and 2025.

### Original Project (2020)
The original project focused on:
- Processing news headlines to track synthetic Zika-like epidemics
- Extracting locations and temporal information
- Generating public health surveillance dashboards
- Using traditional NLP techniques with spaCy and scikit-learn

### Current Enhancement (2025)
We're enhancing the project by:
- Reconstructing the synthetic dataset using modern LLM techniques
- Comparing traditional vs. LLM-generated synthetic data quality
- Implementing parallel analysis pipelines for comparison
- Exploring improvements in NLP and visualization techniques

## Replication Results (2025)

### Data Processing Pipeline
Successfully replicated the core analysis pipeline with modern tools:
- **Data Loading**: 200 LLM-generated headlines (Jan 1 - Jul 18, 2024)
- **Entity Extraction**: 168 locations (84%) and 154 diseases (77%) identified
- **Geolocation**: All locations successfully geocoded using Nominatim
- **Spatial Clustering**: DBSCAN identified 13 distinct outbreak clusters

### Key Findings
1. **Geographic Distribution**: 
   - Southeast Asia: Jakarta (27), Bangkok (17), Manila, Singapore
   - Americas: Miami (18), San Juan, Bogotá, Rio de Janeiro
   - No European or Middle Eastern outbreaks in dataset

2. **Disease Patterns**:
   - Zika virus: 54 cases (35%)
   - TB: 49 cases (32%)
   - Other diseases: 51 cases (33%)

3. **Cluster Analysis**:
   - 13 spatial clusters identified using DBSCAN (eps=400km, min_samples=3)
   - Zero noise points suggests clustering parameters may be too lenient
   - Clusters primarily represent geographic proximity rather than epidemiological relationships

### Contrast with Original Analysis (2020)

| Aspect | Original (2020) | Replication (2025) |
|--------|-----------------|-------------------|
| **Data Source** | Template-based synthetic headlines | LLM-generated headlines |
| **Geographic Coverage** | Global distribution including Europe/Middle East | Limited to tropical/subtropical regions |
| **Disease Types** | Zika-focused with diverse secondary diseases | Zika, TB, and other tropical diseases |
| **Clustering Approach** | KMeans + DBSCAN with custom distance metrics | DBSCAN with geodesic distance |
| **Visualization** | Static Basemap plots | Interactive Folium maps |
| **Analysis Depth** | Multi-phase pipeline with temporal analysis | Streamlined single-script approach |

### Limitations Identified
1. **Geographic Bias**: Current dataset lacks European/Middle Eastern outbreaks
2. **Clustering Interpretation**: Clusters based purely on spatial proximity, not epidemiological relationships
3. **Temporal Analysis**: Limited time-series analysis compared to original
4. **Disease Diversity**: Focus on tropical diseases, missing global health threats

## Project Structure
```
my-ddo/
├── data/                    # Generated synthetic headlines
│   ├── template_headlines.txt  # Basic template-generated headlines
│   └── llm_headlines.txt      # LLM-enhanced headlines
├── 2020Analysis/           # Original analysis notebooks
│   └── step*.ipynb         # Original implementation steps
├── step4_llm_analysis.py   # Replication analysis script
├── outbreak_map.html       # Interactive visualization
├── generate_synthetic_headlines.py  # Template-based generator
└── generate_llm_headlines.py        # LLM-based generator
```

## Recent Changes
1. Data Generation Enhancement:
   - Created two synthetic data generators:
     - Template-based for baseline comparison
     - LLM-based for improved quality
   - Generated parallel datasets for comparative analysis

2. Project Restructuring:
   - Archived original 2020 analysis in `2020Analysis/`
   - Created dedicated `data/` directory for synthetic datasets
   - Set up new analysis pipeline structure

3. Replication Implementation:
   - Successfully replicated core analysis pipeline
   - Generated interactive map visualizations
   - Identified key differences from original approach

## Dependencies
- Python 3.9+
- Core packages: pandas, spaCy, scikit-learn, matplotlib/Seaborn, GeoPy
- Additional requirements in `requirements.txt` and `environment.yml`

## Next Steps
- Implement comparative analysis notebooks
- Evaluate data quality metrics between approaches
- Enhance visualization and dashboard components
- Document improvements and insights
- Address geographic bias in data generation
- Improve clustering interpretation with epidemiological context

## Getting Started
1. Clone the repository
2. Create virtual environment: `python -m venv my-ddo-env`
3. Install dependencies: `pip install -r requirements.txt`
4. Generate synthetic data using provided scripts
5. Run analysis notebooks for comparison

## License
[Original Project License] 