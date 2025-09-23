import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, io, mesh
import basix.ufl
import ufl
import pyvista
import os

# -------------------------------
# 1. Funcție pentru curățarea liniilor
# -------------------------------
def clean_line(line: str):
    """Elimină comentariile după # și întoarce lista de token-uri"""
    return line.split("#")[0].strip().split()

# -------------------------------
# 2. Citește mesh-ul COMSOL
# -------------------------------
def load_comsol_mphtxt(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Fișierul {filename} nu a fost găsit în directorul curent: {os.getcwd()}")

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
            count = int(clean_line(lines[i+2])[0])
            for k in range(count):
                parts = clean_line(lines[i+3+k])
                if len(parts) >= nnodes:
                    try:
                        edges.append(list(map(int, parts[:nnodes])))
                    except ValueError:
                        pass
            i += 3 + count
            continue

        # --- Elemente tri ---
        if line.startswith("3 tri"):
            nnodes = int(clean_line(lines[i+1])[0])
            count = int(clean_line(lines[i+2])[0])
            for k in range(count):
                parts = clean_line(lines[i+3+k])
                if len(parts) >= nnodes:
                    try:
                        triangles.append(list(map(int, parts[:nnodes])))
                    except ValueError:
                        pass
            i += 3 + count
            continue

        i += 1

    return np.array(nodes), np.array(triangles, dtype=int), np.array(edges, dtype=int)

# === MAIN ===
# filename = "comsol2dfara_spire_1pe8_vechi1.mphtxt"
filename = "comsol2dfara_spire_toata_vechi1.mphtxt"

nodes, tris, edgs = load_comsol_mphtxt(filename)

print("Nodes:", nodes.shape)
print("Triangles:", tris.shape)
print("Edges:", edgs.shape)

# Dacă indexarea e 1-based în COMSOL
if tris.min() == 1:
    tris -= 1
if edgs.min() == 1:
    edgs -= 1

# -------------------------------
# 3. Vizualizare rapidă (opțional)
# -------------------------------
plt.figure()
plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)
plt.scatter(nodes[:,0], nodes[:,1], s=5, color="red")
plt.gca().set_aspect("equal")
plt.show()

# -------------------------------
# 4. Creează mesh-ul în FEniCSx
# -------------------------------
points = np.hstack([nodes, np.zeros((nodes.shape[0], 1), dtype=np.float64)])  # 3D cu z=0
cells = tris.astype(np.int64)  # Tip int64

# Definim elementul de coordonate
element = basix.ufl.element("Lagrange", "triangle", degree=1)

# Creează mesh-ul
domain = mesh.create_mesh(MPI.COMM_WORLD, cells, points, element)
print(domain)

# -------------------------------
# 5. Definirea spațiului de funcții
# -------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
print(f"Spațiu de funcții: {V}")

# -------------------------------
# 6. Definirea condițiilor de frontieră
# -------------------------------
fdim = domain.topology.dim - 1

def boundary_left(x):
    return np.isclose(x[0], np.min(nodes[:,0]), rtol=1e-2)

def boundary_right(x):
    return np.isclose(x[0], np.max(nodes[:,0]), rtol=1e-2)

facets_left = mesh.locate_entities_boundary(domain, fdim, boundary_left)
facets_right = mesh.locate_entities_boundary(domain, fdim, boundary_right)

dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
dofs_right = fem.locate_dofs_topological(V, fdim, facets_right)

bc_left = fem.dirichletbc(np.float64(1.0), dofs_left, V)
bc_right = fem.dirichletbc(np.float64(0.0), dofs_right, V)
bcs = [bc_left, bc_right]

# -------------------------------
# 7. Formularea variațională
# -------------------------------
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
eps = fem.Constant(domain, np.float64(1.0))
a = eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, np.float64(0.0)) * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "ilu"})
uh = problem.solve()

# -------------------------------
# 8. Salvează soluția
# -------------------------------
with io.XDMFFile(MPI.COMM_WORLD, "comsol_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

print("Soluția a fost salvată în 'comsol_solution.xdmf'.")

# -------------------------------
# 9. Vizualizare cu PyVista
# -------------------------------
try:
    pyvista.OFF_SCREEN = True
    pyvista.start_xvfb()
    print("Starting visualization with PyVista...")

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.screenshot("comsol_mesh.png")
    print("Mesh salvat în 'comsol_mesh.png'")

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")

    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True, show_scalar_bar=True)
    u_plotter.view_xy()
    u_plotter.screenshot("comsol_solution.png")
    print("Soluția salvată în 'comsol_solution.png'")

except Exception as e:
    print(f"Eroare la vizualizare cu PyVista: {str(e)}")