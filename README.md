# ML Pipeline Automation

## Overview
An automated ML pipeline that:
- Preprocesses the Iris dataset
- Trains a RandomForestClassifier
- Runs unit tests
- Saves the model as an artifact
- Publishes models to GitHub Packages (optional)

## Workflow
1. **Tests**: 
   - Data loading/preprocessing (2 tests)
   - Model accuracy > 90% (1 test)
2. **Training**: 
   - Saves model + scaler as joblib files
3. **Artifacts**: 
   - Uploaded to GitHub Actions
4. **Registry** (Optional):
   - Publishes to GitHub Packages

## How to Use

### Basic Usage (Artifacts)
1. Clone the repo:
   ```bash
   git clone https://github.com/Mahnoor19/ml-pipeline.git
