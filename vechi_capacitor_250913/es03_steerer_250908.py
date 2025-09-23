import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, io, mesh
import basix.ufl
import ufl
import pyvista

# -------------------------------
# 1. Citește mesh-ul COMSOL
# -------------------------------
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
                parts = line.split("#")[0].strip().split()
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
            nnodes = int(lines[i+1].split()[0])
            count = int(lines[i+2].split()[0])
            for k in range(count):
                parts = lines[i+3+k].split()
                if len(parts) >= nnodes:
                    edges.append(list(map(int, parts[:nnodes])))
            i += 3 + count
            continue

        # --- Elemente tri ---
        if line.startswith("3 tri"):
            nnodes = int(lines[i+1].split()[0])
            count = int(lines[i+2].split()[0])
            for k in range(count):
                parts = lines[i+3+k].split()
                if len(parts) >= nnodes:
                    triangles.append(list(map(int, parts[:nnodes])))
            i += 3 + count
            continue

        i += 1

    return np.array(nodes), np.array(triangles, dtype=int), np.array(edges, dtype=int)

# === MAIN ===
nodes, tris, edgs = load_comsol_mphtxt("comsol2dfara_spire_1pe8_vechi1.mphtxt")

print("Nodes:", nodes.shape)
print("Triangles:", tris.shape)
print("Edges:", edgs.shape)

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

# -------------------------------
# 3. Creează mesh-ul în FEniCSx
# -------------------------------
# Adaugă coordonata z=0 pentru a face nodurile 3D (necesar pentru dolfinx)
points = np.hstack([nodes, np.zeros((nodes.shape[0], 1), dtype=np.float64)])  # 3D cu z=0
cells = tris.astype(np.int64)  # Asigură-te că tipul este int64

# Definim elementul de coordonate
element = basix.ufl.element("Lagrange", "triangle", degree=1)  # Element Lagrange de grad 1

# Creează mesh-ul
domain = mesh.create_mesh(MPI.COMM_WORLD, cells, points, element)

# Afișează mesh-ul
print(domain)

# -------------------------------
# 4. Definirea spațiului de funcții
# -------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
print(f"Spațiu de funcții: {V}")

# -------------------------------
# 5. Definirea condițiilor de frontieră
# -------------------------------
fdim = domain.topology.dim - 1  # Dimensiunea fațetelor (1 pentru 2D)

# Definim condiții de frontieră geometrice (exemplu simplu)
def boundary_left(x):
    return np.isclose(x[0], np.min(nodes[:,0]), rtol=1e-2)  # Marginea stângă

def boundary_right(x):
    return np.isclose(x[0], np.max(nodes[:,0]), rtol=1e-2)  # Marginea dreaptă

# Localizează fațetele pentru condițiile de frontieră
facets_left = mesh.locate_entities_boundary(domain, fdim, boundary_left)
facets_right = mesh.locate_entities_boundary(domain, fdim, boundary_right)

# Localizează gradele de libertate
dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
dofs_right = fem.locate_dofs_topological(V, fdim, facets_right)

# Definim condițiile de frontieră Dirichlet
bc_left = fem.dirichletbc(np.float64(1.0), dofs_left, V)  # u=1 pe marginea stângă
bc_right = fem.dirichletbc(np.float64(0.0), dofs_right, V)  # u=0 pe marginea dreaptă
bcs = [bc_left, bc_right]

# -------------------------------
# 6. Formularea variațională
# -------------------------------
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
eps = fem.Constant(domain, np.float64(1.0))  # Constanta dielectrică
a = eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Forma biliniară
L = fem.Constant(domain, np.float64(0.0)) * v * ufl.dx  # Forma liniară

# Rezolva problema
problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "ilu"})
uh = problem.solve()

# -------------------------------
# 7. Salvează soluția
# -------------------------------
with io.XDMFFile(MPI.COMM_WORLD, "comsol_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

print("Soluția a fost salvată în 'comsol_solution.xdmf'.")

# -------------------------------
# 8. Vizualizare cu PyVista
# -------------------------------
try:
    pyvista.OFF_SCREEN = True  # For headless rendering
    pyvista.start_xvfb()  # Necesary on Linux
    print("Starting visualization with PyVista...")

    # Creează mesh VTK pentru domeniu
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Visualizează mesh-ul
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.screenshot("comsol_mesh.png")
    print("Mesh salvat în 'comsol_mesh.png'")

    # Creează mesh VTK pentru soluție
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")

    # Visualizează soluția
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True, show_scalar_bar=True)
    u_plotter.view_xy()
    u_plotter.screenshot("comsol_solution.png")
    print("Soluția salvată în 'comsol_solution.png'")

except Exception as e:
    print(f"Eroare la vizualizare cu PyVista: {str(e)}")