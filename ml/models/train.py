

### File: train.py

from serp_model import SERPModel

def train_models(df):
    print("Training models...")
    serp_model = SERPModel(df)
    lr_model = serp_model.train_linear_regression()
    mlp_model = serp_model.train_mlp_classifier()
    print("Training complete. Models ready!")
    return lr_model, mlp_model