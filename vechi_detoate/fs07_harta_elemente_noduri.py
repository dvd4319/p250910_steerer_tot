import numpy as np
import matplotlib.pyplot as plt
##########################################

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from matplotlib.collections import PolyCollection
###############################################

def clean_line(line: str):
    """Elimină comentariile după # și întoarce lista de token-uri"""
    return line.split("#")[0].strip().split()

############################################################################
################ (0) PRIMA FUNCITE ########################################
############################################################################


def load_comsol_mphtxt(filename):
    nodes = []
    triangles = []
    edges = []
    tri_domains = []

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
            count = int(clean_line(lines[i+2])[0])
            i += 3
            k = 0
            while k < count and i + k < len(lines):
                parts = clean_line(lines[i+k])
                if len(parts) >= nnodes:
                    try:
                        edges.append(list(map(int, parts[:nnodes])))
                    except ValueError:
                        pass
                k += 1
            i += k
            continue

        # --- Elemente tri ---
        if line.startswith("3 tri"):
            nnodes = int(clean_line(lines[i+1])[0])
            count = int(clean_line(lines[i+2])[0])
            i += 3
            k = 0
            while k < count and i + k < len(lines):
                parts = clean_line(lines[i+k])
                if len(parts) >= nnodes:
                    try:
                        triangles.append(list(map(int, parts[:nnodes])))
                    except ValueError:
                        pass
                k += 1
            i += k
            # --- Citim domeniile triunghiurilor ---
            while i < len(lines):
                parts = clean_line(lines[i])
                if parts and parts[0].isdigit():
                    try:
                        count_domains = int(parts[0])
                        i += 1
                        k = 0
                        while k < count_domains and i + k < len(lines):
                            parts = clean_line(lines[i+k])
                            if parts and parts[0].isdigit():
                                try:
                                    tri_domains.append(int(parts[0]))
                                except ValueError:
                                    pass
                            k += 1
                        i += k
                        break
                    except ValueError:
                        pass
                i += 1
            continue

        i += 1

    return np.array(nodes), np.array(triangles, dtype=int), np.array(edges, dtype=int), np.array(tri_domains, dtype=int)
############################################################################
################ (1) PRIMA FIGURA -- mphtxt ####################################
############################################################################
# # === MAIN ===
# nodes, tris, edgs, tri_domains = load_comsol_mphtxt("comsol2dfara_spire_1pe8_vechi1_491_492.mphtxt")
nodes, tris, edgs, tri_domains = load_comsol_mphtxt("comsol2dfara_spire_toata_vechi1.mphtxt")

# print("Nodes:", nodes.shape)
# print("Triangles:", tris.shape)
# print("Edges:", edgs.shape)

# plt.figure(figsize=(8,8))

# if tris.size > 0:
#     plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

# if edgs.size > 0:
#     for e in edgs:
#         x = nodes[e,0]
#         y = nodes[e,1]
#         plt.plot(x, y, color="red")

        

# plt.scatter(nodes[:,0], nodes[:,1], s=5, color="black")
# plt.gca().set_aspect("equal")
# plt.title("FIG. (1) Mesh din COMSOL .mphtxt")
# plt.show()
####################################################
###########

###########################################################################
############### (2) A DOUA FIGURA -- mphtxt cu noduri numerotate ##########
###########################################################################

# plt.figure(figsize=(8,8))

# # --- Desenează triunghiurile albastre ---
# if tris.size > 0:
#     plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.1)

# # --- Desenează liniile roșii pentru edge-uri ---
# if edgs.size > 0:
#     for e in edgs:
#         x = nodes[e,0]
#         y = nodes[e,1]
#         plt.plot(x, y, color="red", linewidth=0.5)

# # --- Desenează nodurile ---
# plt.scatter(nodes[:,0], nodes[:,1], s=2, color="black")

# # --- Numerotează nodurile ---
# for i, (x, y) in enumerate(nodes):
#     plt.text(x, y, str(i), color="black", fontsize=10, ha="center", va="center")

# # --- Aspect și titlu ---
# plt.gca().set_aspect("equal")
# plt.title("FIG. (2) Mesh din COMSOL .mphtxt cu noduri numerotate și linii roșii")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.show()

###########################################################################
############### (3) A DOUA FUNCITE ########################################
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

def build_adjacency(tris, edgs):
    # Facem set de muchii de frontieră (cele roșii)
    boundary_edges = {tuple(sorted(e)) for e in edgs}

    # Construim graful de adiacență între triunghiuri
    adjacency = defaultdict(list)
    edge_to_tri = defaultdict(list)

    for t_idx, tri in enumerate(tris):
        for k in range(3):
            e = tuple(sorted((tri[k], tri[(k+1)%3])))
            edge_to_tri[e].append(t_idx)

    for e, tlist in edge_to_tri.items():
        if len(tlist) == 2 and e not in boundary_edges:
            # triunghiurile împart muchia și nu e frontieră
            t1, t2 = tlist
            adjacency[t1].append(t2)
            adjacency[t2].append(t1)

    return adjacency

###########################################################################
############### (4) A TREIA FUNCITE ########################################
###########################################################################
def find_domains(tris, adjacency):
    visited = np.zeros(len(tris), dtype=bool)
    domains = np.full(len(tris), -1, dtype=int)
    domain_id = 0

    for t in range(len(tris)):
        if not visited[t]:
            # BFS/DFS pentru componenta conexă
            queue = deque([t])
            visited[t] = True
            domains[t] = domain_id
            while queue:
                cur = queue.popleft()
                for neigh in adjacency[cur]:
                    if not visited[neigh]:
                        visited[neigh] = True
                        domains[neigh] = domain_id
                        queue.append(neigh)
            domain_id += 1
    return domains
##############################################################################################################
# # === MAIN ===
adj = build_adjacency(tris, edgs)
domains = find_domains(tris, adj)

print(f"Număr domenii detectate: {domains.max()+1}")

plt.figure(figsize=(8,8))
plt.tripcolor(nodes[:,0], nodes[:,1], tris, facecolors=domains, cmap="tab10", edgecolors="k", linewidth=0.3)

# Adăugăm și muchiile de frontieră (roșii)
for e in edgs:
    x = nodes[e,0]
    y = nodes[e,1]
    plt.plot(x, y, color="red", linewidth=1.2)

plt.gca().set_aspect("equal")
plt.title("FIG. (3)  Domenii identificate automat")
plt.colorbar(label="Domain ID")
plt.show()

# ##############################################################################################################

# adj = build_adjacency(tris, edgs)
# domains = find_domains(tris, adj)

# print(f"Număr domenii detectate: {domains.max()+1}")
# ### ===================================== ### 

# # Construim poligoanele din triunghiuri
# polys = [nodes[tri] for tri in tris]


# lista_culori = ["skyblue", "gray", "orange", "red", "green", "yellow"]
# # Convertim culorile din text în RGBA numeric
# import matplotlib.colors as mcolors
# face_colors = [mcolors.to_rgba(c) for c in lista_culori]  # lista_culori e lista ta ["skyblue", "gray", "orange", ...]

# # Plotăm
# fig, ax = plt.subplots(figsize=(8,8))
# pc = PolyCollection(polys, facecolors=face_colors, edgecolors="k", linewidths=0.3)
# ax.add_collection(pc)

# # Setări
# ax.autoscale_view()
# ax.set_aspect("equal")
# plt.title("FIG. (4) Mesh colorat pe domenii")
# plt.show()

# # ##############################################################################################################
# # Mapăm domeniile la materiale
# domain_material = {
#     "D1": "Aer",
#     "D2": "Fier",
#     "D3": "Cupru",
#     "D4": "Aer",
#     "D5": "Fier"
# }

# # Alocăm câte o culoare pentru fiecare material
# material_colors = {
#     "Aer": "skyblue",
#     "Fier": "gray",
#     "Cupru": "orange"
# }

# plt.figure(figsize=(8,8))

# # Colorăm triunghiurile în funcție de material
# face_colors = []
# for t in range(len(tris)):
#     # verificăm în ce domeniu cade triunghiul `t`
#     d = domains[t]
#     if d == 0: mat = "Aer"
#     elif d == 1: mat = "Fier"
#     elif d == 2: mat = "Cupru"
#     elif d == 3: mat = "Aer"
#     elif d == 4: mat = "Fier"
#     else: mat = "Aer"  # fallback
    
#     face_colors.append(material_colors[mat])

# ##############################################################################################################
# # edgs are forma (numar_muchii, 2)
# edge_nodes = np.unique(edgs.flatten())

# print("Nodurile aflate pe muchii:")
# print(edge_nodes)

# edge_coords = nodes[edge_nodes]

# for idx, (x, y) in zip(edge_nodes, edge_coords):
#    print(f"Nod {idx}: ({x:.3f}, {y:.3f})")

# plt.figure(figsize=(8,8))
# plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

# # #toate nodurile
# plt.scatter(nodes[:,0], nodes[:,1], s=5, color="black", label="Noduri")

# # # doar nodurile de pe muchii
# plt.scatter(edge_coords[:,0], edge_coords[:,1], s=30, color="red", label="Noduri pe muchii")

# plt.gca().set_aspect("equal")
# plt.legend()
# plt.title("FIG. (5) Nodurile de pe muchii evidențiate")
# plt.show()

############################################################################


# def find_domains1(nodes, edges):
#     """Găsește domenii închise (poligoane) pe baza muchiilor"""
#     G = nx.Graph()
#     G.add_edges_from([tuple(e) for e in edges])

#     # căutăm cicluri simple
#     cycles = nx.cycle_basis(G)

#     # convertim în poligoane
#     polygons = []
#     for cycle in cycles:
#         coords = [nodes[i] for i in cycle]
#         poly = Polygon(coords)
#         if poly.is_valid and poly.area > 1e-12:
#             polygons.append(poly)

#     # sortăm după arie
#     polygons = sorted(polygons, key=lambda p: p.area, reverse=True)

#     # primul (cel mai mare) = domeniul exterior -> îl ignorăm
#     domains = polygons[1:]
#     return domains
# ########################################################################################################################
# # # === MAIN ===

# domains = find_domains1(nodes, edgs)
# print(f"Am detectat {len(domains)} domenii interioare.")

# # plot
# plt.figure(figsize=(8, 8))
# plt.triplot(nodes[:,0], nodes[:,1], tris, color="lightgray", linewidth=0.5)

# colors = ["red", "green", "blue", "orange", "purple", "cyan"]

# for i, poly in enumerate(domains):
#     x, y = poly.exterior.xy
#     plt.fill(x, y, alpha=0.4, color=colors[i % len(colors)], label=f"Domeniu {i+1}")

# plt.scatter(nodes[:,0], nodes[:,1], s=5, color="black")
# plt.gca().set_aspect("equal")
# plt.legend()
# plt.title("FIG. (7) Domenii detectate automat din muchii")
# plt.show()
