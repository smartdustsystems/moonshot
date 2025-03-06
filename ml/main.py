
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from serp_model import SERPModel

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

    print("Training models...")
    serp_model = SERPModel(df)
    lr_model = serp_model.train_linear_regression()
    mlp_model = serp_model.train_mlp_classifier()

    print("Training complete. Models ready!")
