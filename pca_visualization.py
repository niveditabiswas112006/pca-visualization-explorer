import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Load a high-dimensional dataset (Digits dataset: 64 features)
    print("Loading the Digits dataset...")
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    print(f"Original dataset shape: {X.shape} (64 dimensions)")

    # It's generally a good practice to scale the data before applying PCA
    # so that features with larger scales do not dominate the principal components.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply PCA to reduce to 2 components
    print("\nApplying PCA to reduce data to 2 components...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Reduced dataset shape: {X_pca.shape} (2 dimensions)")

    # 3. Report how much variance is retained
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance_retained = sum(explained_variance_ratio) * 100

    print(f"\n--- Variance Retained ---")
    print(f"Explained variance by Principal Component 1: {explained_variance_ratio[0]*100:.2f}%")
    print(f"Explained variance by Principal Component 2: {explained_variance_ratio[1]*100:.2f}%")
    print(f"Total variance retained by 2 components: {total_variance_retained:.2f}%")

    print("\n--- What does this mean in practice? ---")
    print(f"By reducing the dimensions from {X.shape[1]} to 2, we are able to visualize the data in a simple 2D plot.")
    print(f"However, we only retain {total_variance_retained:.2f}% of the original information (variance) in the dataset.")
    print("This means there's a trade-off: we gain the ability to easily visualize and simplify the data,")
    print(f"but we lose {100 - total_variance_retained:.2f}% of the information. Some digit classes might overlap in 2D because")
    print("the features distinguishing them might be captured in the components we discarded.")
    print("----------------------------------------\n")

    # 4. Visualize the reduced data
    print("Generating 2D visualization plot...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, edgecolor='k')

    # Add a colorbar and labels
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Class (0-9)')
    
    plt.title('PCA of Digits Dataset (Reduced to 2D)')
    plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]*100:.2f}% variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]*100:.2f}% variance)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    plot_filename = 'pca_2d_visualization.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved 2D visualization plot to '{plot_filename}'")
    
    # Uncomment the line below if you want to display the plot interactively when running the script
    # plt.show()

if __name__ == "__main__":
    main()
