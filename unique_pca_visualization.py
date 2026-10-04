import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ssl

def run_custom_pca_analysis():
    # ---------------------------------------------------------
    # STEP 1: Load a unique high-dimensional dataset
    # ---------------------------------------------------------
    # To avoid common datasets (like Iris, Digits, or Breast Cancer) 
    # we are fetching the "Sonar" dataset directly from the UCI Machine Learning Repository.
    # It contains 60 features representing sonar signals bouncing off Rocks (R) or Metal Mines (M).
    
    # Fix for macOS Python SSL certificate issues when downloading from URLs
    ssl._create_default_https_context = ssl._create_unverified_context
    
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    print(f"Fetching Sonar dataset from: {dataset_url}")
    
    # The dataset has no header, 60 features, and the 61st column is the label
    df = pd.read_csv(dataset_url, header=None)
    
    # Extract features (first 60 columns) and target (last column)
    feature_data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    print(f"\nSuccessfully loaded! Dataset shape: {feature_data.shape} (60 dimensions)")
    
    # ---------------------------------------------------------
    # STEP 2: Preprocess and apply PCA
    # ---------------------------------------------------------
    # Standardizing features to have zero mean and unit variance is crucial for PCA
    standardizer = StandardScaler()
    features_scaled = standardizer.fit_transform(feature_data)
    
    print("Applying Principal Component Analysis (PCA) to reduce 60 dimensions to 2...")
    pca_model = PCA(n_components=2)
    reduced_features = pca_model.fit_transform(features_scaled)
    
    print(f"Shape after reduction: {reduced_features.shape}")

    # ---------------------------------------------------------
    # STEP 3: Variance retention report
    # ---------------------------------------------------------
    var_ratio = pca_model.explained_variance_ratio_
    total_var = np.sum(var_ratio) * 100

    print("\n================ VARIANCE REPORT ================")
    print(f"Variance captured by PC1: {var_ratio[0]*100:.2f}%")
    print(f"Variance captured by PC2: {var_ratio[1]*100:.2f}%")
    print(f"Total variance retained : {total_var:.2f}%")
    print("=================================================")
    
    print("\n[Practical Interpretation]")
    print(f"We have compressed 60 distinct features of sonar data into just 2 dimensions.")
    print(f"By doing this, we keep {total_var:.2f}% of the original patterns and information.")
    print(f"While we lose roughly {100 - total_var:.2f}% of the variance, we gain the ability to")
    print(f"visually plot and inspect the relationship between Mines and Rocks on a flat screen.")

    # ---------------------------------------------------------
    # STEP 4: 2D Visualization
    # ---------------------------------------------------------
    print("\nGenerating scatter plot for the reduced data...")
    plt.figure(figsize=(9, 7))
    
    # We have two classes: 'M' (Mine) and 'R' (Rock)
    unique_classes = np.unique(labels)
    colors = ['crimson', 'dodgerblue']
    
    for cls, col in zip(unique_classes, colors):
        # Filter rows belonging to the current class
        idx = labels == cls
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], 
                    c=col, label=f"Class: {cls}", alpha=0.75, edgecolors='white', s=60)

    plt.title('2D PCA Projection of the 60-Dimensional Sonar Dataset', fontsize=14, pad=15)
    plt.xlabel(f'Principal Component 1 ({var_ratio[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({var_ratio[1]*100:.1f}%)', fontsize=12)
    plt.legend(title='Target Label')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save the output visualization
    output_img = 'unique_pca_visualization.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization successfully saved as '{output_img}'")
    
if __name__ == "__main__":
    run_custom_pca_analysis()
