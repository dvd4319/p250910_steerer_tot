import numpy as np
import matplotlib.pyplot as plt
#############################################
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#################################################################
def clean_line(line: str):
    """Elimină comentariile după # și întoarce lista de token-uri"""
    return line.split("#")[0].strip().split()

def load_comsol_mphtxt(filename):
    nodes = []
    triangles = []
    edges = []

    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Secțiunea noduri ---
        if line.startswith("# Mesh point coordinates"):
            i += 1
            while i < len(lines) and not lines[i].startswith("#"):
                parts = clean_line(lines[i])
                if len(parts) >= 2:
                    try:
                        x, y = map(float, parts[:2])
                        nodes.append((x, y))
                    except ValueError:
                        pass
                i += 1
            continue

        # --- Elemente edg ---
        if line.startswith("3 edg"):
            nnodes = int(clean_line(lines[i+1])[0])
            count  = int(clean_line(lines[i+2])[0])
            for k in range(count):
                parts = clean_line(lines[i+3+k])
                if len(parts) >= nnodes:
                    edges.append(list(map(int, parts[:nnodes])))
            i += 3 + count
            continue

        # --- Elemente tri ---
        if line.startswith("3 tri"):
            nnodes = int(clean_line(lines[i+1])[0])
            count  = int(clean_line(lines[i+2])[0])
            for k in range(count):
                parts = clean_line(lines[i+3+k])
                if len(parts) >= nnodes:
                    triangles.append(list(map(int, parts[:nnodes])))
            i += 3 + count
            continue

        i += 1

    return np.array(nodes), np.array(triangles, dtype=int), np.array(edges, dtype=int)


# === MAIN ===
# filename = "comsol2dfara_spire_1pe8_vechi1.mphtxt"
filename = "comsol2dfara_spire_toata_vechi1.mphtxt"

nodes, tris, edgs = load_comsol_mphtxt(filename)

# Dacă indexarea e 1-based în COMSOL, trecem la 0-based
if tris.min() == 1:
    tris -= 1
if edgs.min() == 1:
    edgs -= 1

print("Nodes:", nodes.shape)
print("Triangles:", tris.shape)
print("Edges:", edgs.shape)

# --- Definirea subdomeniilor prin nodurile lor ---
D1 = [90, 73, 71, 89]  # Aer1
D2 = [364, 380, 383, 401, 403, 404, 425, 453, 472, 488, 492, 491, 481, 464, 441, 433, 410, 432, 431, 430, 418, 447, 448, 449, 420, 400, 381]  # Fier1
D3 = [428, 426, 407, 389, 372, 353, 337, 323, 309, 294, 277, 258, 255, 256, 257, 272, 287, 304, 318, 330, 343, 361, 378, 395, 412, 435, 443, 465, 477, 479, 478, 473, 455, 428]  # Cupru
D4 = [404, 391, 387, 373, 356, 354, 339, 325, 311, 296, 278, 259, 241, 238, 235, 236, 210, 177, 138, 133, 101, 103, 100, 99, 108, 121, 124, 125, 123, 114, 85, 74, 62, 52, 44, 36, 25, 13, 4, 2, 1, 0, 7, 6, 5, 15, 28, 40, 51, 61, 70, 82, 98, 132, 169, 199, 228, 250, 269, 286, 303, 317, 329, 342, 358, 376, 392, 410, 433, 441, 464, 481, 491, 492, 488, 472, 453, 425]  # Aer2
D5 = [85, 114, 123, 125, 124, 121, 108, 99, 100, 103, 101, 133, 138, 177, 210, 236, 235, 238, 241, 259, 278, 296, 311, 325, 339, 354, 356, 373, 387, 391, 404, 403, 401, 383, 380, 364, 363, 348, 347, 333, 320, 315, 302, 295, 282, 265, 246, 222, 191, 158, 120, 90, 89, 71, 83, 86, 87, 88]  # Fier2
###############################################################################################
# subdomains = [D1, D2, D3, D4, D5]
# colors = ["cyan", "orange", "red", "green", "magenta"]

# plt.figure(figsize=(10,10))
# plt.gca().set_aspect("equal")

# # --- Plotează triunghiurile ---
# plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

# # --- Plotează muchiile (roșii) ---
# for e in edgs:
#     x = nodes[e,0]
#     y = nodes[e,1]
#     plt.plot(x, y, color="red", linewidth=1)

# # --- Evidențiază subdomeniile prin conturul lor ---
# for i, sub in enumerate(subdomains):
#     sub_coords = nodes[sub, :]
#     # Închide conturul
#     if not np.array_equal(sub_coords[0], sub_coords[-1]):
#         sub_coords = np.vstack([sub_coords, sub_coords[0]])
#     plt.plot(sub_coords[:,0], sub_coords[:,1], color=colors[i], linewidth=2, label=f"D{i+1}")

# # --- Noduri ---
# plt.scatter(nodes[:,0], nodes[:,1], s=10, color="black")

# plt.title("Mesh COMSOL cu subdomenii evidențiate")
# plt.legend()
# plt.show()
# #######################################################################################
# --- Datele subdomeniilor ---
subdomains = [D1, D2, D3, D4, D5]
colors = ["green", "magenta", "red", "green", "magenta"]
# labels = ["Aer1", "Fier1", "Cupru", "Aer2", "Fier2"]
labels = ["Air1", "Iron1", "Copper", "Air2", "Iron2"]


plt.figure(figsize=(10,10))
ax = plt.gca()
ax.set_aspect("equal")

# --- Plotează triunghiurile ---
plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

# --- Plotează muchiile (roșii) ---
for e in edgs:
    x = nodes[e,0]
    y = nodes[e,1]
    plt.plot(x, y, color="red", linewidth=1)

# --- Creează și adaugă poligoanele subdomeniilor ---
patches = []
for sub, color, label in zip(subdomains, colors, labels):
    coords = nodes[sub, :]
    # Închide poligonul
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    poly = Polygon(coords, closed=True)
    patches.append(poly)

p = PatchCollection(patches, facecolor=colors, alpha=0.3, edgecolor="k", linewidths=1.5)
ax.add_collection(p)

# --- Adaugă etichetele materialelor în centrul fiecărui subdomeniu ---
for sub, label in zip(subdomains, labels):
    coords = nodes[sub, :]
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    centroid = coords.mean(axis=0)
    ax.text(centroid[0], centroid[1], label, ha="center", va="center", fontsize=20, alpha=0.9)

# --- Nodurile ---
plt.scatter(nodes[:,0], nodes[:,1], s=10, color="black")

# plt.title("Mesh cu subdomenii colorate și etichete")
plt.title("Mesh with Colored Subdomains and Labels")
plt.show()

########################################################
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np

# subdomeniile, fiecare ca listă de indici noduri
subdomains = [D1, D2, D3, D4, D5]

polygons = []
for sub in subdomains:
    coords = nodes[sub, :]
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    polygons.append(Polygon(coords))

# unirea tuturor subdomeniilor
domain_union = unary_union(polygons)

# exteriorul domeniului mare
external_coords = np.array(domain_union.exterior.coords)

# afișare
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.plot(external_coords[:,0], external_coords[:,1], 'r-', linewidth=2)  # frontiera externă
plt.scatter(nodes[:,0], nodes[:,1], s=10, color='black')
plt.gca().set_aspect('equal')
plt.title("External boundary of the full domain")
plt.show()

#################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Creează un array cu toate muchiile triunghiurilor
edges = np.vstack([tris[:, [0,1]],
                   tris[:, [1,2]],
                   tris[:, [2,0]]])

# Sortează nodurile fiecărei muchii (ca să identificăm duplicate)
edges = np.sort(edges, axis=1)

# Numărăm aparițiile fiecărei muchii
edges_tuple = [tuple(e) for e in edges]
unique_edges, counts = np.unique(edges_tuple, return_counts=True, axis=0)

# Frontieră = muchii care apar o singură dată
boundary_edges = np.array([e for e,c in zip(unique_edges, counts) if c==1])

# Dirichlet: y ≈ 0
dirichlet_edges = [e for e in boundary_edges if np.all(np.abs(nodes[e,1]) < 1e-12)]
dirichlet_edges = np.array(dirichlet_edges)

# Neumann: restul frontierelor
neumann_edges = np.array([e for e in boundary_edges if not np.all(np.abs(nodes[e,1]) < 1e-12)])

# --- Plot ---
plt.figure(figsize=(8,8))
plt.gca().set_aspect('equal')

plt.triplot(nodes[:,0], nodes[:,1], tris, color="lightblue", linewidth=0.5)

# Dirichlet (roșu)
for e in dirichlet_edges:
    plt.plot(nodes[e,0], nodes[e,1], 'r', linewidth=2)

# Neumann (verde)
for e in neumann_edges:
    plt.plot(nodes[e,0], nodes[e,1], 'g', linewidth=2)

plt.title("Frontiere Dirichlet (roșu) și Neumann (verde)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

####################################################################################
