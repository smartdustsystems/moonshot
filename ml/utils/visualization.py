import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()
