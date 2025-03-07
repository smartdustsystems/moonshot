

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.train import train_models
from utils.visualization import plot_correlation_matrix

if __name__ == "__main__":
    print("Loading data...")
    data_loader = DataLoader(num_samples=20)
    data_loader.generate_synthetic_data()
    data_loader.enrich_with_html()
    df = data_loader.get_data()

    print("Extracting features...")
    feature_engineer = FeatureEngineer(df)
    feature_engineer.apply_html_extraction()
    feature_engineer.extract_spacy_embeddings()
    df = feature_engineer.get_features()

    plot_correlation_matrix(df)
    
    lr_model, mlp_model = train_models(df)