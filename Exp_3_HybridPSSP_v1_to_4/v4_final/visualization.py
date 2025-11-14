"""
visualization.py
----------------
Visualize confusion matrices for Q3 and Q8 protein secondary structure predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 - kept for quick style tweaks during experiments
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Hard-coded confusion matrices captured from the best v4 checkpoint logs.
# These stay in source control so teammates can reproduce the figure without rerunning inference.
# Q8 Confusion Matrix
q8_cm = np.array([
    [63982,   317,  2141,  1233,    73,   764,     3,     0],
    [  346, 39095,  4526,   502,   454,    84,    68,     0],
    [ 2701,  4121, 40267,  2500,  2070,   503,    83,     0],
    [ 3089,   342,  3253, 14481,   720,   895,     5,     0],
    [  702,   740,  6641,  2712,  5200,   290,    31,     0],
    [ 1630,   176,  1422,  1537,   107,  3257,     2,     0],
    [   80,   491,  1235,   169,   102,    25,   222,     0],
    [   30,     1,     2,     9,     5,     1,     0,     0]
])

# Q3 Confusion Matrix  
q3_cm = np.array([
    [69240,   400,  7052],
    [  446, 39362,  7591],
    [ 7424,  4757, 79165]
])

# Q8 Class Labels
q8_labels = ['H', 'E', 'C', 'T', 'S', 'G', 'B', 'I']

# Q3 Class Labels
q3_labels = ['H', 'E', 'C']

# Create the visualization canvas with one subplot per task.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("HybridPSSP v4 Confusion Matrices", fontsize=16, fontweight='bold')

# Plot Q8 Confusion Matrix
im1 = ax1.imshow(q8_cm, cmap='Blues', aspect='auto')
ax1.set_xticks(np.arange(len(q8_labels)))
ax1.set_yticks(np.arange(len(q8_labels)))
ax1.set_xticklabels(q8_labels)
ax1.set_yticklabels(q8_labels)
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
ax1.set_title('Q8 Secondary Structure Confusion Matrix\n(8-class prediction)', fontsize=14, fontweight='bold')

# Add text annotations for Q8
for i in range(len(q8_labels)):
    for j in range(len(q8_labels)):
        if q8_cm[i, j] > 1000:  # Only show text for significant values
            ax1.text(j, i, f'{q8_cm[i, j]:,}', 
                    ha="center", va="center", 
                    color="white" if q8_cm[i, j] > np.max(q8_cm)/2 else "black",
                    fontsize=9)

# Plot Q3 Confusion Matrix
im2 = ax2.imshow(q3_cm, cmap='Blues', aspect='auto')
ax2.set_xticks(np.arange(len(q3_labels)))
ax2.set_yticks(np.arange(len(q3_labels)))
ax2.set_xticklabels(q3_labels)
ax2.set_yticklabels(q3_labels)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
ax2.set_title('Q3 Secondary Structure Confusion Matrix\n(3-class prediction)', fontsize=14, fontweight='bold')

# Add text annotations for Q3
for i in range(len(q3_labels)):
    for j in range(len(q3_labels)):
        ax2.text(j, i, f'{q3_cm[i, j]:,}', 
                ha="center", va="center", 
                color="white" if q3_cm[i, j] > np.max(q3_cm)/2 else "black",
                fontsize=11)

# Add colorbars
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Print aggregate statistics so the console output mirrors the plots.
print("Q3 CONFUSION MATRIX ANALYSIS:")
print(f"Total predictions: {np.sum(q3_cm):,}")
print(f"Correct predictions: {np.trace(q3_cm):,}")
print(f"Overall accuracy: {np.trace(q3_cm)/np.sum(q3_cm):.3f}")

print("\nQ8 CONFUSION MATRIX ANALYSIS:")
print(f"Total predictions: {np.sum(q8_cm):,}")
print(f"Correct predictions: {np.trace(q8_cm):,}")
print(f"Overall accuracy: {np.trace(q8_cm)/np.sum(q8_cm):.3f}")

print("\nQ8 Class Distribution:")
for i, label in enumerate(q8_labels):
    total = np.sum(q8_cm[i, :])
    correct = q8_cm[i, i]
    accuracy = correct / total if total > 0 else 0
    print(f"  {label}: {correct:>6,} / {total:>6,} = {accuracy:.3f}")

print("\nQ3 Class Distribution:")
for i, label in enumerate(q3_labels):
    total = np.sum(q3_cm[i, :])
    correct = q3_cm[i, i]
    accuracy = correct / total if total > 0 else 0
    print(f"  {label}: {correct:>6,} / {total:>6,} = {accuracy:.3f}")