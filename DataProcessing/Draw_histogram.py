import os
import matplotlib.pyplot as plt
from collections import Counter

# Category mapping
id_to_name = {
    28: 'suitcase',
    0: 'person',
    60: 'table',
    56: 'chair'
}

labels_dir = '. /labels_new' # Replace with your labels folder path

# Initialize the counter
counter = Counter()

# Iterate over all labeled files
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(labels_dir, filename), 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                cls_id = int(line.strip().split()[0])
                counter[cls_id] += 1

# Ensure that mapping is done in categorical order 
labels = [id_to_name[i] for i in sorted(id_to_name.keys())]
counts = [counter[i] for i in sorted(id_to_name.keys())]

# draw
plt.figure(figsize=(8, 5))

plt.bar(labels, counts, color='cornflowerblue')
plt.title('Class Distribution in Dataset')

plt.xlabel('Object Class')
plt.ylabel('Number of Instances')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
