import matplotlib.pyplot as plt
import pandas as pd

# Construct data
data_6x4 = {
    "Optimizer": ["AdamW", "SGD", "SGD + Momentum"],
    "Avg Loss": [0.0612, 0.5181, 0.1442],
    "Precision": [0.8711, 0.7536, 0.8689],
    "Recall": [0.9348, 0.9054, 0.9369],
    "Remarks": [
        "Fast convergence,\nstrong on regression",
        "Slow,\nweak performance",
        "Balanced performance,\nstrong generalization"
    ],
}

# Create the DataFrame and set the Optimizer to the index
df_6x4 = pd.DataFrame(data_6x4)
df_6x4.set_index("Optimizer", inplace=True)
df_6x4 = df_6x4.transpose() # transpose to 6 rows × 3 columns (plus column headers for a total of 4 columns)

# Draw the table
plt.figure(figsize=(9, 4.5)) # width x height in inches
plt.axis('off') # cancel axis display

# Create form objects
tbl = plt.table(
    cellText=df_6x4.values,
    colLabels=df_6x4.columns,
    rowLabels=df_6x4.index,
    cellLoc='center',
    rowLoc='center',
    loc='center',
    colColours=["#d9eaf7"] * df_6x4.shape[1]  # Top column header color
)

# Setting font size and scaling
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.4)

# Save as high resolution image
plt.tight_layout()
plt.savefig("optimizer_comparison_6x4.png", dpi=300)
plt.show()
