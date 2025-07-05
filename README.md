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

## Project Structure
```
my-ddo/
├── data/                    # Generated synthetic headlines
│   ├── template_headlines.txt  # Basic template-generated headlines
│   └── llm_headlines.txt      # LLM-enhanced headlines
├── 2020Analysis/           # Original analysis notebooks
│   └── step*.ipynb         # Original implementation steps
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

## Dependencies
- Python 3.9+
- Core packages: pandas, spaCy, scikit-learn, matplotlib/Seaborn, GeoPy
- Additional requirements in `requirements.txt` and `environment.yml`

## Next Steps
- Implement comparative analysis notebooks
- Evaluate data quality metrics between approaches
- Enhance visualization and dashboard components
- Document improvements and insights

## Getting Started
1. Clone the repository
2. Create virtual environment: `python -m venv my-ddo-env`
3. Install dependencies: `pip install -r requirements.txt`
4. Generate synthetic data using provided scripts
5. Run analysis notebooks for comparison

## License
[Original Project License] 