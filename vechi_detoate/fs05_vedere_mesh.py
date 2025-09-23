# import meshio
# import matplotlib.pyplot as plt

# # mesh = meshio.read("mesh_valid.msh")
# mesh = meshio.read("comsol2dfara_spire_1pe8_vechi1_fixed.msh")


# points = mesh.points
# cells = mesh.cells_dict["triangle"]  # sau "line" / "tetra" după caz

# plt.triplot(points[:,0], points[:,1], cells)
# plt.gca().set_aspect("equal")
# plt.show()
###################################################################################

# with open("comsol2dfara_spire_1pe8_vechi1.mphtxt") as f:
#     for i, line in enumerate(f):
#         print(f"{i+1:03d}: {line.strip()}")
#         if i > 50:
#             break

#################################################################################
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

        # linia cu numărul de noduri: "<int> # number of mesh points"
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

if nodes.size > 0:
    plt.scatter(nodes[:,0], nodes[:,1], s=5)
    plt.gca().set_aspect("equal")
    plt.show()
else:
    print("Nu am găsit noduri în fișier!")
