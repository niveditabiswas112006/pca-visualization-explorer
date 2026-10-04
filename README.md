# PCA Visualization Explorer

This repository contains a Python implementation of Principal Component Analysis (PCA) for reducing the dimensionality of high-dimensional datasets to 2D for visualization.

## Overview
Dimensionality reduction is a crucial tool for visualization and simplification, allowing us to reason about the trade-off between simplicity and information loss.

In this project, we fetch the **Sonar Dataset** (60 dimensions) directly from the UCI Machine Learning Repository to demonstrate PCA. By reducing the data from 60 dimensions to 2, we are able to visualize the separation between the two classes (Mines vs. Rocks) on a 2D scatter plot.

## Files
- `unique_pca_visualization.py`: The main script that fetches the Sonar dataset, applies PCA, calculates variance retention, and generates the plot.
- `unique_pca_visualization.png`: The output 2D scatter plot showing the data after dimensionality reduction.

## Output Details
When the script is run, it reports the variance retained by the top 2 principal components. While we lose some variance by dropping 58 dimensions, we gain the ability to visualize complex, high-dimensional patterns in a simple 2D space.
