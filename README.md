# RecSys - IMDb Recommender System

Recommender system for IMDb dataset with ML pipeline, feature engineering, and deployment infrastructure.

## Project Structure

```
RecSys/
├── src/                          # Source code
│   ├── data_loader.py           # Data loading
│   ├── preprocessor.py          # Data preprocessing
│   ├── feature_engineer.py      # Feature engineering
│   ├── feature_store.py         # Feature storage
│   ├── config.py                # Configuration
│   └── pipeline/                # Pipeline scripts
│       └── preprocess_pipeline.py
├── data/                         # Data directory
│   ├── raw/movielens_datasets/   # Raw movielens data
│   └── processed/feature_store/  # Processed features
├── requirements.txt              # Dependencies
└── Makefile                      # Automation commands
```

## Quick Start

### 1. Install Dependencies

```bash
make install
```

### 2. Run Preprocessing Pipeline

```bash
make preprocess
```

### 3. Clean Up

```bash
make clean
```

## Pipeline Overview

### Data Loading

- Loads IMDb CSV datasets
- Filters relevant columns for efficiency
- Handles missing values

### Preprocessing

- Filters titles by minimum votes (100+)
- Filters by year (1990+)
- Merges multiple datasets
- Aggregates principal cast/crew

### Feature Engineering

- **Genre features**: Binary indicators for top genres
- **Rating features**: Weighted ratings, vote categories
- **Temporal features**: Title age, decade, era
- **Content features**: Runtime categories, title length
- **Popularity features**: Popularity score, engagement levels

### Feature Store

- Saves features in Parquet format
- Stores metadata and statistics
- Versioned storage for reproducibility

## Configuration

Edit `src/config.py` to customize:

- Minimum vote threshold
- Year range
- Column selections
- File paths

## Python Environment

Using: `/Users/ivan/myvenv`
