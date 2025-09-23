import numpy as np
import matplotlib.pyplot as plt

nodes = []


# with open("comsol2dfara_spire_1pe8_vechi1.mphtxt") as f:
with open("comsol2dfara_spire_toata_vechi1.mphtxt") as f:
    n_nodes = 0
    reading_nodes = False
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "Mesh point coordinates" in line:
                reading_nodes = True
            continue

        if line.endswith("# number of mesh points"):
            n_nodes = int(line.split()[0])
            continue

        if reading_nodes and len(nodes) < n_nodes:
            parts = line.split()
            if len(parts) >= 2:
                x, y = map(float, parts[:2])
                nodes.append((x, y))

nodes = np.array(nodes)
print("Nodes shape:", nodes.shape)

# Vizualizare noduri cu etichete
plt.figure(figsize=(6, 6))
plt.scatter(nodes[:,0], nodes[:,1], s=10, color="blue", alpha=0.6)

# dacă vrei numerotarea nodurilor
for i, (x, y) in enumerate(nodes):
    plt.text(x, y, str(i), fontsize=6, color="red")

plt.gca().set_aspect("equal")
plt.title("Nodurile din mesh (500 puncte)")
plt.show()
