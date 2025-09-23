from fs15_01_nods_edg_tri_plot import (f01_load_comsol_mphtxt, f02_build_adjacency, f03_find_domains, f04_convert_mphtxt_to_gmsh, f04_convert_mphtxt_to_gmsh_nou, f05_load_comsol_msh, f06_convert_msh_to_xdmf, f07_extract_triangles_to_xdmf, f08_inspect_xdmf_mesh, f09_read_xdmf_mesh)
from fs15_01_nods_edg_tri_plot import (p01_plot_mesh_mphtxt,  p02_plot_mesh_with_labels, p03_plot_domains_mphtxt, 
                                       p04_plot_cell_tags, p05_plot_domains_gmesh1, p06_visualize_xdmf_mesh, p07_plot_subdomains)
from fs15_01_nods_edg_tri_plot import ( p08_plot_external_boundary, p09_plot_dirichlet_neumann_boundaries, p10_plot_subdomains_tris)
########################################################################################################
from fs15_solve_problem import assign_material_properties, define_boundary_conditions
from fs15_solve_problem import assign_materials, plot_materials_on_mesh, assign_materials_variant, plot_materials_on_mesh_variant
########################################################################################################
from dolfinx import mesh, fem, plot,default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import read_from_msh, model_to_mesh
from dolfinx.mesh import locate_entities_boundary, compute_midpoints, meshtags
from dolfinx.fem import dirichletbc, Expression, Function, functionspace, locate_dofs_topological, form, locate_dofs_geometrical 
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, LinearProblem
from dolfinx.plot import vtk_mesh
########################################################################################################
from ufl import TestFunction, TrialFunction, as_vector, dot, dx, grad, inner, Measure 
from petsc4py import PETSc
########################################################################################################
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista
import pyvista as pv
import ufl 
import meshio
import basix.ufl
import h5py
import basix
##########################################################
##########################################################
rank = MPI.COMM_WORLD.rank
gdim = 2
model_rank = 0
mesh_comm = MPI.COMM_WORLD
MPI.COMM_WORLD.barrier()
comm = MPI.COMM_WORLD
############################################################################

# funcție pentru maparea valorilor domeniilor pe noduri
def map_cell_tags_to_nodes(mesh, cell_tags, values_dict):
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cell_to_nodes = mesh.topology.connectivity(tdim, 0)
    
    num_nodes = mesh.topology.index_map(0).size_local
    num_cells = mesh.topology.index_map(tdim).size_local
    node_values = np.zeros(num_nodes, dtype=np.float64)
    
    node_domain_counts = [[] for _ in range(num_nodes)]
    for cell_idx in range(num_cells):
        nodes = cell_to_nodes.links(cell_idx)
        domain = cell_tags.values[cell_idx]
        for node in nodes:
            node_domain_counts[node].append(domain)
    
    for node in range(num_nodes):
        if node_domain_counts[node]:
            domain = max(set(node_domain_counts[node]), key=node_domain_counts[node].count)
            node_values[node] = values_dict[domain]
    
    return node_values
# #############################################################
# ### ================= MPHTXT ============================ ###
# #############################################################
mphtxt_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403.mphtxt"
nodes_mphtxt, tris_mphtxt, edgs_mphtxt, tri_domains_mphtxt = f01_load_comsol_mphtxt(mphtxt_file)
print("###############################################")
print("DATELE SUNT CULESTE DIN .mphtxt")
print("## ======================================= ##")
# print("Nodes mphtxt:", nodes.shape)
print("Triangles mphtxt:", tris_mphtxt.shape)
print("Edges mphtxt:", edgs_mphtxt.shape)
print("Tri domains from mphtxt:", tri_domains_mphtxt.shape)
# print("Edge domains from mphtxt:", edg_domains)
print("## ======================================= ##")
print("###############################################")

adj_mphtxt = f02_build_adjacency(tris_mphtxt, edgs_mphtxt)
domains_mphtxt = f03_find_domains(tris_mphtxt, adj_mphtxt)

# print("###############################################")
# print("DATELE SUNT CULESTE DIN .mphtxt")
# print("## ======================================= ##")
# print("adj mphtxt:", adj)
# print("domains mphtxt:", domains)
# print("## ======================================= ##")
# print("###############################################")

p01_plot_mesh_mphtxt(nodes_mphtxt, tris_mphtxt, edgs_mphtxt, title="Fig. (1) Mesh: 2D section - format .mphtxt")
p02_plot_mesh_with_labels(nodes_mphtxt, tris_mphtxt, edgs_mphtxt, title="Fig. (2) Mesh with numbered nodes and red lines indicating the domain boundaries")

# Exemplu: definim manual materialele pe domenii
domain_materials_mphtxt = {
    0: "0 IRON",
    1: "1 AIR",
    2: "2 COPPER",
}

# Culori definite manual (în aceeași ordine ca ID-urile)
domain_colors_mphtxt = {
    0: "#646363",  # Iron → maro cupru
    1: "#314B7A",  # Air 
    2: "#B87333",  # Copper → gri
}

domain_label_pos_mphtxt = {
    0: (-0.15, 0.10),  # domeniul 0: x=0.1, y=0.2
    1: (-0.05, 0.03),
    2: (-0.18, 0.04)
}

p03_plot_domains_mphtxt(nodes_mphtxt, tris_mphtxt, edgs_mphtxt, domains_mphtxt, domain_materials_mphtxt, domain_colors_mphtxt, title="Fig. (3) domains - format .mphtxt",domain_label_pos=domain_label_pos_mphtxt)

# ###########################

# # #############################################################
# # ### ================= mphtxt IN msh ===================== ###
# # #############################################################
msh_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_nou.msh" ### creeaza msh (inital gol)
# tri_domains_corect = f04_convert_mphtxt_to_gmsh(msh_file) ## tranforma in msh si afiseaza 
tri_domains_corect = f04_convert_mphtxt_to_gmsh_nou(mphtxt_file, msh_file) ## tranforma in msh si afiseaza 
nodes_msh, triangles_msh, edges_msh = f05_load_comsol_msh(msh_file) ### incarca msh ca sa fie vazut 

print("###############################################")
print("DATELE SUNT CULESTE DIN .msh")
print("## ======================================= ##")
print("Nodes msh:", nodes_msh.shape)
print("Triangles msh:", triangles_msh.shape)
print("Edges msh:", edges_msh.shape)
print("Triangle correct msh: ", tri_domains_corect.shape)
print("Triangle nr correct msh: ", len(tri_domains_corect))
#print("Tri domains from mphtxt:", tri_domains)
print("## ======================================= ##")
print("###############################################")

###########################
domain_materials_3 = {
    1: "1 IRON",
    2: "2 AIR",
    3: "3 COPPER",
}
domain_colors_3 = {
    1: "#726B6B",  # Iron → maro cupru
    2: "#154A86",  # Air 
    3: "#B9761E",  # Copper → gri
}
domain_label_pos_3 = {
    1: (-0.15, 0.10),  # domeniul 0: x=0.1, y=0.2
    2: (-0.05, 0.03),
    3: (-0.18, 0.04)
}

p05_plot_domains_gmesh1(nodes_msh, triangles_msh, edges_msh, tri_domains_corect,domain_colors_3, domain_materials_3,
                           title="Fig. (5) 2D section - from format NOU .msh",domain_label_pos = domain_label_pos_3)
# # print("Tri domains corect from msh:", tri_domains_corect)
# # # # # #############################################################
# # # # # ### ================= msh IN xdmf ======================= ###
# # # # # #############################################################

# Citește fișierul .msh
msh_to_xdmf = meshio.read(msh_file)

edge_cells_xdmf = None
for cell_block in msh_to_xdmf.cells:
    if cell_block.type == "line":
        edge_cells_xdmf = cell_block.data
        break

if edge_cells_xdmf is None:
    raise ValueError("Nu s-au găsit celule de tip 'edge' în fișierul .msh")

if edge_cells_xdmf is None:
    print("⚠️ Nu am găsit muchii în fișierul .xdmf/.msh")
else:
    print(f"Am găsit {len(edge_cells_xdmf)} muchii")


# Extrage celulele de tip "triangle"
triangle_cells_xdmf = None
for cell_block in msh_to_xdmf.cells:
    if cell_block.type == "triangle":
        triangle_cells_xdmf = cell_block.data
        break

if triangle_cells_xdmf is None:
    raise ValueError("Nu s-au găsit celule de tip 'triangle' în fișierul .msh")

print(f" lungime triangle_cells_xdmf = {len(triangle_cells_xdmf)}")

# tri_domains_corect
# # Extrage datele despre domenii (gmsh:physical) pentru triunghiuri
triangle_domains_xdmf = None
if "gmsh:physical" in msh_to_xdmf.cell_data:
    for i, cell_block in enumerate(msh_to_xdmf.cells):
        if cell_block.type == "triangle":
            triangle_domains_xdmf = msh_to_xdmf .cell_data["gmsh:physical"][i]
            break

import numpy as np


if triangle_domains_xdmf is None:
    raise ValueError("Nu s-au găsit date despre domenii (gmsh:physical) pentru triunghiuri")

print("###############################################")
print("DATELE SUNT CULESTE DIN .msh")
print("## ======================================= ##")
# Verifică lungimea și tipul datelor
print(f"Număr triunghiuri xdmf: {triangle_cells_xdmf.shape}")
print(f"Număr domenii xdmf: {triangle_domains_xdmf.shape}")
print(f"Tip triangle_domains xdmf: {type(triangle_domains_xdmf)}")
# print(f"Primele 10 domenii: {triangle_domains_xdmf}")
print("## ======================================= ##")
print("###############################################")

##############################################################################################
##############################################################################################
##############################################################################################
# după ce citești din XDMF
nodes_xdmf = msh_to_xdmf.points[:, :2]  # doar primele 2 coordonate
tris_xdmf = triangle_cells_xdmf
tri_domains_xdmf = triangle_domains_xdmf

# culori și materiale (le poți refolosi pe cele de la .msh)
domain_materials_xdmf = {
    1: "1 IRON",
    2: "2 AIR",
    3: "3 COPPER",
}
domain_colors_xdmf = {
    1: "#918282",
    2: "#2C76A1",
    3: "#9AA04D",
}
domain_label_pos_xdmf = {
    1: (-0.15, 0.10),
    2: (-0.05, 0.03),
    3: (-0.18, 0.04),
}

# # plotează pt xdmf 
# p05_plot_domains_gmesh1(
#     nodes_xdmf,
#     tris_xdmf,
#     edge_cells_xdmf if edge_cells_xdmf is not None else np.zeros((0, 2)),
#     tri_domains_xdmf,
#     domain_colors_xdmf,
#     domain_materials_xdmf,
#     title="Fig. (X) Domenii din XDMF",
#     domain_label_pos=domain_label_pos_xdmf
# )

#####################################################################
# === Scriere .xdmf doar triunghiuri (2D) cu domenii ===
#####################################################################
xdmf_tri_file = msh_file.replace(".msh", "_triangles.xdmf")
meshio.write(
    xdmf_tri_file,
    meshio.Mesh(
        points=msh_to_xdmf.points,
        cells=[("triangle", triangle_cells_xdmf)],
        cell_data={"gmsh:physical": [triangle_domains_xdmf.astype(np.int32)]}
    )
)
print(f"✅ Fișier XDMF triunghiuri scris: {xdmf_tri_file}")

#####################################################################
# === Scriere .xdmf doar muchii (1D boundary) ===
#####################################################################
if edge_cells_xdmf is not None and len(edge_cells_xdmf) > 0:
    xdmf_edge_file = msh_file.replace(".msh", "_edges.xdmf")
    meshio.write(
        xdmf_edge_file,
        meshio.Mesh(
            points=msh_to_xdmf.points,
            cells=[("line", edge_cells_xdmf)],
            cell_data={"gmsh:physical": [np.zeros(len(edge_cells_xdmf), dtype=np.int32)]}
        )
    )
    print(f"✅ Fișier XDMF muchii scris: {xdmf_edge_file}")

#####################################################################
# === Citire mesh 2D și domenii în DolfinX ===
#####################################################################
with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")  # domeniile
######################################################################


# --- Citire mesh 2D și domenii ---
xdmf_tri_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_nou_triangles.xdmf"
with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")  # domeniile

# --- Extrage nodurile și triunghiurile pentru plot ---
nodes_xdmf_1 = mesh.geometry.x[:, :2]
triangles_xdmf_1 = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)
tri_domains_xdmf_1 = ct.values

# --- Culori și materiale (poți refolosi ce ai definit) ---
domain_colors_xdmf_1 = {1: "#442A2A", 2: "#78A12C", 3: "#26C4C4"}
domain_materials_xdmf_1 = {1: "IRON", 2: "AIR", 3: "COPPER"}
domain_label_pos_xdmf_1 = {1: (-0.15, 0.10), 2: (-0.05, 0.03), 3: (-0.18, 0.04)}

# --- Plot mesh și domenii ---
# p05_plot_domains_gmesh1(
#     nodes=nodes_xdmf_1,
#     triangles=triangles_xdmf_1,
#     edges=np.zeros((0, 2)),  # dacă nu ai muchii
#     tri_domains_corect=tri_domains_xdmf_1,
#     domain_colors=domain_colors_xdmf_1,
#     domain_materials=domain_materials_xdmf_1,
#     title=" dupa brambureala -- Mesh din XDMF",
#     domain_label_pos=domain_label_pos_xdmf_1
# )


# citește mesh-ul și meshtags din XDMF (triunghiuri + domenii)
with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cell_tags = xdmf.read_meshtags(mesh, name="Grid")  # aici sunt domeniile reale

# extrage coordonate noduri și conectivitate triunghiuri
cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)
points = mesh.geometry.x

# culori pentru fiecare domeniu
colors = cell_tags.values

# plot mesh colorat pe domenii
plt.tripcolor(points[:, 0], points[:, 1], cells, facecolors=colors,
              cmap="tab20", edgecolors="k", linewidth=0.2)
plt.gca().set_aspect("equal")
plt.colorbar(label="Domenii")
plt.title("Mesh colorat pe domenii (din XDMF)")
plt.show()

################################################################################################

# Citește mesh-ul și domeniile din XDMF
with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cell_tags = xdmf.read_meshtags(mesh, name="Grid")  # domeniile

# Noduri 2D → transformăm în 3D pentru PyVista
points_raw = mesh.geometry.x            # Nx4 (sau Nx2/3)
points_2d = points_raw[:, :2]           # doar x și y
points_3d = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])  # Nx3, z=0

# Extrage conectivitatea triunghiurilor
tdim = mesh.topology.dim
cells_connectivity = mesh.topology.connectivity(tdim, 0)
cells = np.array([cells_connectivity.links(i) for i in range(mesh.topology.index_map(tdim).size_local)])

# Construim array-ul VTK pentru UnstructuredGrid: [3, i0, i1, i2, ...]
cells_vtk = np.hstack([np.full((cells.shape[0], 1), 3), cells]).flatten()

# Tipurile celulelor
cell_types = np.full(cells.shape[0], pv.CellType.TRIANGLE)

# Creăm PyVista UnstructuredGrid
grid = pv.UnstructuredGrid(cells_vtk, cell_types, points_3d)

# Setăm scalarul pentru domenii
grid.cell_data["domain"] = cell_tags.values

# Vizualizare PyVista
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, scalars="domain", cmap="tab20", show_scalar_bar=True)
plotter.add_text("Mesh colorat pe domenii - XDMF - PY-VISTA", position="upper_edge", font_size=14, color="black")
plotter.view_xy()
plotter.show_axes()   # afișează axele X, Y, Z
plotter.view_xy()     # vedeți mesh-ul din planul XY
plotter.add_axes()
plotter.show()

##############################################################################################
with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
    domain_2= xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain_2, name="Grid")   # <- very important#

dx = ufl.Measure("dx", domain=domain_2, subdomain_data=ct)

VV = functionspace(domain_2, ("CG", 1))  # continuous space
uu = TrialFunction(VV)
vv = TestFunction(VV)

mu0 = 4.0 * np.pi * 1e-7
mu_Fe = 4000*mu0
mu_val = mu_Fe
J_val = 3.4e6

a = (1/mu_val) * dot(grad(uu), grad(vv)) * dx(1) \
  + (1/mu0) * dot(grad(uu), grad(vv)) * dx(2) \
  + (1/mu0) * dot(grad(uu), grad(vv)) * dx(3)

L = J_val * vv * dx(3)   # doar în subdomeniul cu id=3

tdim = domain_2.topology.dim
fdim = tdim-1
dirichlet_facets = locate_entities_boundary(domain_2, fdim, lambda x: ~np.isclose(x[1], 0.0))  # Dirichlet doar pe frontiera unde y != 0
dofs = locate_dofs_topological(VV, fdim, dirichlet_facets)
bc = dirichletbc(default_scalar_type(0), dofs, VV)

A_z = Function(VV)
problem = LinearProblem(a, L, u=A_z, bcs=[bc])
problem.solve()
#########################################################################################

# Salvează soluția A_z în fișier XDMF
with XDMFFile(MPI.COMM_WORLD, "A_z_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain_2)
    xdmf.write_function(A_z)

import numpy as np
np.save("Az_values.npy", A_z.x.array)
# # -----------------------------
# # Crează PyVista grid pentru A_z
# # -----------------------------
# pv.start_xvfb()  # necesară pe Linux dacă nu există display
# Az_grid = pv.UnstructuredGrid(domain_2._cpp_object)  # conversia plasei dolfinx în pyvista
Az_grid = pv.UnstructuredGrid(cells_vtk, cell_types, points_3d)
Az_grid.point_data["A_z"] = A_z.x.array  # adaugă valorile lui A_z la noduri
Az_grid.set_active_scalars("A_z") # setează A_z ca scalar activ
Az_grid.cell_data["domain"] = cell_tags.values

# # -----------------------------
# # Vizualizare PyVista
# # -----------------------------
plotter = pv.Plotter()

# Fundal: mesh colorat după domenii
plotter.add_mesh(
    grid,
    scalars="domain",
    cmap="tab20",
    show_edges=True,
    opacity=1.0,
    show_scalar_bar=True
)

# Suprapunere: câmp A_z
plotter.add_mesh(
    Az_grid,
    scalars="A_z",
    cmap="Reds",
    show_edges=True,
    scalar_bar_args={"title":"A_z"},
    opacity=0.6
)

# Setăm planul XY și axele
plotter.view_xy()
plotter.camera.parallel_projection = True
# plotter.add_title("Magnetic Induction B from curl(A_z)", font_size=14)
plotter.show_grid()
# Afișare
plotter.show()
### *******************

###########################################################################
########### CALCULEAZA B -varianta 1 - ####################################
###########################################################################

##########################################################################
# --- 1. Spațiu de funcții vectorial DG0 (discontinuu pe celule) ---
dim = 2  # dimensiune spațiu fizic
with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   # <- foarte important#

dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
W = functionspace(domain, ("Discontinuous Lagrange", 0, (dim,)))
B = Function(W)

V_scalar = functionspace(domain, ("CG", 1))
A_z = Function(V_scalar)
A_z.x.array[:] = np.load("Az_values.npy")
# Expression UFL for curl(A_z) în 2D: B = (∂A_z/∂y, -∂A_z/∂x)
B_expr = as_vector([A_z.dx(1), -A_z.dx(0)])
# Trial și test pentru W (vector DG0)
u_proj = TrialFunction(W)
v_proj = TestFunction(W)
# Forme pentru L2 projection: ∫ u_proj * v_proj dx = ∫ B_expr * v_proj dx
a_proj = inner(u_proj, v_proj) * dx  # bilinear (masă)
L_proj = inner(B_expr, v_proj) * dx  # linear (right-hand side cu curl)

# Rezolvă projection (fără BC-uri, DG0 nu are; solver local per celulă)
problem_proj = LinearProblem(a_proj, L_proj, u=B)  # u=B se umple cu soluția
problem_proj.solve()
############################################################################
########### PLOTEAZA B -varianta CU SAGETI  #############################
###########################################################################

B_grid = pv.UnstructuredGrid(*vtk_mesh(mesh))
B_values = B.x.array.reshape(-1, 2)
B_mag = np.linalg.norm(B_values, axis=1)
B_grid.cell_data["|B|"] = B_mag
B_grid.set_active_scalars("|B|")

# # Săgeți normalizate pentru direcție
cell_centers = B_grid.cell_centers()
vectors_cell = np.column_stack([B_values[:, 0], B_values[:, 1], np.zeros(len(B_values))])
norms = np.linalg.norm(vectors_cell[:, :2], axis=1, keepdims=True)
norms[norms == 0] = 1.0
B_norm = vectors_cell / norms
cell_centers.cell_data.clear()
cell_centers.cell_data["Bnorm"] = B_norm

# # Factor lungime săgeți
domain_size = max(B_grid.bounds[1]-B_grid.bounds[0], B_grid.bounds[3]-B_grid.bounds[2], 1.0)
desired_arrow_length = 0.01 * domain_size
quiver = cell_centers.glyph(orient="Bnorm", scale=False, factor=desired_arrow_length)

# [
#  'viridis', 'plasma', 'inferno', 'magma', 'cividis',   # sequential perceptually uniform
#  'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu',
#  'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
#  'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
#  'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
#  'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
#  'twilight', 'twilight_shifted', 'hsv', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 
#  'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
# ]

# # ================= Vizualizare PyVista =================
plotter = pv.Plotter(window_size=[1200, 900])
# fundal: domenii
plotter.add_mesh(grid, scalars="domain", cmap="tab20", show_edges=True, opacity=0.0,  show_scalar_bar=False)
# B magnitudine
plotter.add_mesh(B_grid, scalars="|B|", cmap="jet", show_edges=True, opacity=1, scalar_bar_args={"title":"|B| (T)"},  show_scalar_bar=False)
# săgeți direcționale
plotter.add_mesh(quiver, color="red", opacity=1, show_scalar_bar=False)

# # Contur domenii
for d in np.unique(cell_tags.values):
    mask = cell_tags.values == d
    cells_d = B_grid.extract_cells(np.where(mask)[0])
    plotter.add_mesh(cells_d, color=None, show_edges=True, edge_color="white", line_width=2, opacity=0)

plotter.view_xy()
plotter.camera.parallel_projection = True
plotter.add_title("Magnetic Induction B from curl(A_z)", font_size=14)
plotter.show_grid()
plotter.show()

#####################################################################################
##################### VALORI NUMERICE ###############################################
#####################################################################################

# # --- Citește mesh și meshtags ---
# with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
#     mesh = xdmf.read_mesh(name="Grid")
#     ct = xdmf.read_meshtags(mesh, name="Grid")

# # Topologie și geometrie
# cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape((-1, 3))
# points = mesh.geometry.x

# # Dacă mesh-ul e 2D, adaugă z=0 pentru PyVista
# if points.shape[1] == 2:
#     points = np.hstack([points, np.zeros((points.shape[0], 1))])

# # Centrii celulelor
# cell_centers = points[cells].mean(axis=1)

# # ID-uri și domenii
# cell_ids = np.arange(cells.shape[0])
# cell_domains = ct.values

# # Creare mesh PyVista (PolyData pentru triunghiuri)
# grid = pv.PolyData()
# grid.points = points
# grid.faces = np.hstack([np.full((cells.shape[0], 1), 3), cells]).astype(np.int64)

# # Adaugă date pentru celule
# grid.cell_data["Cell_ID"] = cell_ids
# grid.cell_data["Domain"] = cell_domains

# # Vizualizare 2D
# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True, scalars="Domain", cmap="tab20")

# # Etichete doar pentru un subset de celule ca să nu blocheze
# subset = np.arange(0, cells.shape[0], max(1, cells.shape[0]//200))  # afișează ~50 etichete
# for i in subset:
#     plotter.add_point_labels([cell_centers[i]], [f"{cell_ids[i]}:{cell_domains[i]}"],
#                              font_size=10, point_color="red", text_color="black")

# # Setare vedere ortogonală 2D
# plotter.view_xy()
# plotter.camera.parallel_projection = False 
# plotter.add_title("Domeniile Numerotate", font_size=14)
# plotter.show_grid()
# plotter.show()


#####################################################################################
##################### VALORI NUMERICE ###############################################
#####################################################################################

cell_index_42 = 42  # celula dorită (aer)
cell_index_67 = 67  # celula dorită (aer)
cell_index_75 = 75  # celula dorită (aer)
cell_index_467 = 467  # celula dorită
# presupunem că B este DG0 și are 2 componente
B_values = B.x.array.reshape((-1, 2))  # fiecare rând = o celulă

print(f"B în celula {cell_index_42}:", B_values[cell_index_42])
print(f"B în celula {cell_index_67}:", B_values[cell_index_67])
print(f"B în celula {cell_index_75}:", B_values[cell_index_75])
###########################################################################
print(f"B în celula {cell_index_467}:", B_values[cell_index_467])


import numpy as np

# Exemplu: B_values pentru toate celulele
# B_values[i] = [B_x, B_y] pentru celula i
B_norm = np.linalg.norm(B_values, axis=1)  # norma euclidiană pe fiecare celulă

# Exemplu: norma pentru celula 42 și 467
print("||B|| în celula 42:", B_norm[42])
print("||B|| în celula 67:", B_norm[67])
print("||B|| în celula 75:", B_norm[75])
###################################################
print("||B|| în celula 467:", B_norm[467])

######################################################################

import numpy as np

# Extrage valorile nodurilor pentru A_z
A_values = A_z.x.array  # valori la noduri

# Dacă vrei media pe celule (echivalent DG0), trebuie să faci medie peste noduri
cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)

def cell_average_values(cell_idx):
    nodes = cells[cell_idx]
    return np.mean(A_values[nodes])

# Celulele de interes
cell_indices = [42, 67, 75, 467]

for idx in cell_indices:
    val = cell_average_values(idx)
    print(f"A_z mediu în celula {idx}:", val)
