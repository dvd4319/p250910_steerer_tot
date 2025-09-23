import numpy as np
import matplotlib.pyplot as plt
#######################################
import numpy as np
import matplotlib.pyplot as plt
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
# from dolfinx.mesh import create_mesh, CellType
########################################
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
# nodes, tris, edgs = load_comsol_mphtxt("comsol2dfara_spire_1pe8_vechi1.mphtxt")
nodes, tris, edgs = load_comsol_mphtxt("comsol2dfara_spire_toata_vechi1.mphtxt")

print("Nodes:", nodes.shape)
print("Triangles:", tris.shape)
print("Edges:", edgs.shape)

plt.figure(figsize=(8,8))

if tris.size > 0:
    plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

if edgs.size > 0:
    for e in edgs:
        x = nodes[e,0]
        y = nodes[e,1]
        plt.plot(x, y, color="red")

plt.scatter(nodes[:,0], nodes[:,1], s=5, color="black")
plt.gca().set_aspect("equal")
plt.title("Mesh din COMSOL .mphtxt")
plt.show()

###########################################################

# -------------------------------
# 1. Citește mesh-ul COMSOL
# -------------------------------
# nodes, tris, edgs = load_comsol_mphtxt("comsol2dfara_spire_1pe8_vechi1.mphtxt")
nodes, tris, edgs = load_comsol_mphtxt("comsol2dfara_spire_toata_vechi1.mphtxt")


# Dacă indexarea e 1-based în COMSOL
if tris.min() == 1:
    tris -= 1
if edgs.min() == 1:
    edgs -= 1

# -------------------------------
# 2. Vizualizare rapidă (opțional)
# -------------------------------
plt.figure()
plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)
plt.scatter(nodes[:,0], nodes[:,1], s=5, color="red")
plt.gca().set_aspect("equal")
plt.show()

# # -------------------------------
# # 3. Crearea mesh-ului FEniCSx
# # -------------------------------
# points = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])  # facem 3D cu z=0
# cells = tris  # deja 0-based
# cell_type = dolfinx.cpp.mesh.CellType.triangle
# mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, cell_type)

# nodes și tris sunt deja 0-based
# points = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])  # adaug z=0
# cells = {"triangle": tris.astype(np.int32)}  # dictionar necesar

# mesh = create_mesh(MPI.COMM_WORLD, cells, points, CellType.triangle)



# # -------------------------------
# # 4. Salvare mesh în XDMF
# # -------------------------------
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_from_comsol.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)