from fs15_01_nods_edg_tri_plot import (f01_load_comsol_mphtxt, f02_build_adjacency, f03_find_domains, f04_convert_mphtxt_to_gmsh, f05_load_comsol_msh, f06_convert_msh_to_xdmf, f07_extract_triangles_to_xdmf, f08_inspect_xdmf_mesh, f09_read_xdmf_mesh)
from fs15_01_nods_edg_tri_plot import (p01_plot_mesh_mphtxt,  p02_plot_mesh_with_labels, p03_plot_domains_mphtxt, 
                                       p04_plot_cell_tags, p05_plot_domains_gmesh1, p06_visualize_xdmf_mesh, p07_plot_subdomains)
from fs15_01_nods_edg_tri_plot import ( p08_plot_external_boundary, p09_plot_dirichlet_neumann_boundaries, p10_plot_subdomains_tris)
########################################################################################################
from vechi_detoate.fs15_solve_problem import assign_material_properties, define_boundary_conditions
from vechi_detoate.fs15_solve_problem import   assign_materials, plot_materials_on_mesh, assign_materials_variant, plot_materials_on_mesh_variant
#################################################################################################
import meshio
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags
from dolfinx.fem import Function, dirichletbc, functionspace, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from ufl import TrialFunction, TestFunction, inner, grad, dx, Measure
import basix.ufl
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_geometrical
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import h5py
import pyvista
from petsc4py import PETSc


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

# funcție pentru corectarea tri_domains
def f04_convert_mphtxt_to_gmsh_nou(mphtxt_file,msh_file):
    nodes, tris, edgs, tri_domains = f01_load_comsol_mphtxt(mphtxt_file)
    tri_domains_correct = tri_domains[3:757]
    # print(f"tri domains INAINTE: {tri_domains}")
    # print(f"tri domains DUPA: {tri_domains_correct}")
    print(f"LUNGIME tri domains INAINTE: {len(tri_domains)}")
    print(f"LUNGIME tri domains DUPA: {len(tri_domains_correct)}")
    if len(tri_domains_correct) != len(tris):
        raise ValueError(f"lungimea tri_domains_correct ({len(tri_domains_correct)}) nu coincide cu lungimea tris ({len(tris)})")
    
    unique_domains = np.unique(tri_domains_correct)
    print(f"valori unice în tri_domains_correct (înainte de corecție): {unique_domains}")
    
    tri_domains_correct = np.where(tri_domains_correct == 0, 1, tri_domains_correct)
    
    unique_domains_corrected = np.unique(tri_domains_correct)
    if not np.all(np.isin(unique_domains_corrected, [1, 2, 3])):
        raise ValueError(f"tri_domains_correct conține valori neașteptate după corecție: {unique_domains_corrected}")
    
    with open(msh_file, "w") as out:
        out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        out.write("$Nodes\n")
        out.write(f"{len(nodes)}\n")
        for i, (x, y) in enumerate(nodes, 1):
            out.write(f"{i} {x} {y} 0\n")
        out.write("$EndNodes\n")
        out.write("$Elements\n")
        out.write(f"{len(tris) + len(edgs)}\n")
        for i, (n1, n2) in enumerate(edgs, 1):
            out.write(f"{i} 1 2 0 0 {n1 + 1} {n2 + 1}\n")
        for i, (t, dom) in enumerate(zip(tris, tri_domains_correct), len(edgs) + 1):
            out.write(f"{i} 2 2 {dom} {dom} {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        out.write("$EndElements\n")
    print(f"număr tri-domenii detectate: {len(unique_domains_corrected)}")
    return tri_domains_correct

# generează fișierul .msh
mphtxt_file="comsol2dfara_spire_1pe8_vechi1_3dom_403.mphtxt"
msh_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403.msh"
tri_domains_corect = f04_convert_mphtxt_to_gmsh_nou(mphtxt_file, msh_file)
#############################################################################################################
# #############################################################
# ### ================= MPHTXT ============================ ###
# #############################################################
nodes, tris, edgs, tri_domains = f01_load_comsol_mphtxt(mphtxt_file)
print("###############################################")
print("DATELE SUNT CULESTE DIN .mphtxt")
print("## ======================================= ##")
# print("Nodes mphtxt:", nodes.shape)
print("Triangles mphtxt:", tris)
print("Edges mphtxt:", edgs.shape)
print("Tri domains from mphtxt:", tri_domains)
# print("Edge domains from mphtxt:", edg_domains)
print("## ======================================= ##")
print("###############################################")

adj = f02_build_adjacency(tris, edgs)
domains = f03_find_domains(tris, adj)
# print("###############################################")
# print("DATELE SUNT CULESTE DIN .mphtxt")
# print("## ======================================= ##")
# print("adj mphtxt:", adj)
# print("domains mphtxt:", domains)
# print("## ======================================= ##")
# print("###############################################")

# p01_plot_mesh_mphtxt(nodes, tris, edgs, title="Fig. (1) Mesh: 2D section - format .mphtxt")
# p02_plot_mesh_with_labels(nodes, tris, edgs, title="Fig. (2) Mesh with numbered nodes and red lines indicating the domain boundaries")

# Exemplu: definim manual materialele pe domenii
domain_materials = {
    0: "0 IRON",
    1: "1 AIR",
    2: "2 COPPER",
}

# Culori definite manual (în aceeași ordine ca ID-urile)
domain_colors = {
    0: "#646363",  # Iron → maro cupru
    1: "#314B7A",  # Air 
    2: "#B87333",  # Copper → gri
}

domain_label_pos = {
    0: (-0.15, 0.10),  # domeniul 0: x=0.1, y=0.2
    1: (-0.05, 0.03),
    2: (-0.18, 0.04)
}

p03_plot_domains_mphtxt(nodes, tris, edgs, domains, domain_materials, domain_colors, title="Fig. (3) domains - format .mphtxt",domain_label_pos=domain_label_pos)
# # #############################################################
# # ### ================= mphtxt IN msh ===================== ###
# # #############################################################
nodes, triangles, edges = f05_load_comsol_msh(msh_file) ### incarca msh ca sa fie vazut 
print("###############################################")
print("DATELE SUNT CULESTE DIN .msh")
print("## ======================================= ##")
print("Nodes msh:", nodes.shape)
print("Triangles msh:", triangles.shape)
print("Edges msh:", edges.shape)
print("Triangle correct msh: ", tri_domains_corect)
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
    1: "#A5A1A1",  # Iron → maro cupru
    2: "#2CA15D",  # Air 
    3: "#CEBB4E",  # Copper → gri
}
domain_label_pos_3 = {
    1: (-0.15, 0.10),  # domeniul 0: x=0.1, y=0.2
    2: (-0.05, 0.03),
    3: (-0.18, 0.04)
}

p05_plot_domains_gmesh1(nodes, triangles, edges, tri_domains_corect,domain_colors_3, domain_materials_3,
                            title="Fig. (5) 2D section - from format .msh",domain_label_pos = domain_label_pos_3)


##############################################################################################################
# citește fișierul .msh
msh = meshio.read(msh_file)

# extrage celulele de tip "triangle"
triangle_cells = None
for cell_block in msh.cells:
    if cell_block.type == "triangle":
        triangle_cells = cell_block.data
        break

if triangle_cells is None:
    raise ValueError("nu s-au găsit celule de tip 'triangle' în fișierul .msh")

# extrage datele despre domenii
triangle_domains = None
if "gmsh:physical" in msh.cell_data:
    for i, cell_block in enumerate(msh.cells):
        if cell_block.type == "triangle":
            triangle_domains = msh.cell_data["gmsh:physical"][i]
            break

if triangle_domains is None:
    raise ValueError("nu s-au găsit date despre domenii (gmsh:physical) pentru triunghiuri")

# verifică lungimea și tipul datelor
print(f"număr triunghiuri: {len(triangle_cells)}")
print(f"număr domenii: {len(triangle_domains)}")
print(f"tip triangle_domains: {type(triangle_domains)}")
print(f"primele 10 domenii: {triangle_domains[:10]}")
print(f"domenii unice (din .msh): {np.unique(triangle_domains)}")

# asigură-te că triangle_domains este un array numpy
triangle_domains = np.asarray(triangle_domains, dtype=np.int32)

# verifică dacă lungimile coincid
if len(triangle_cells) != len(triangle_domains):
    raise ValueError(f"lungimea triangle_cells ({len(triangle_cells)}) nu coincide cu lungimea triangle_domains ({len(triangle_domains)})")

# scrie fișierul .xdmf
xdmf_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.xdmf"
meshio.write(
    xdmf_file,
    meshio.Mesh(
        points=msh.points,
        cells=[("triangle", triangle_cells)],
        cell_data={"domains": [triangle_domains]}
    )
)

# citește plasa în fenicsx
with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
#########################################################################################
# citește datele despre domenii din fișierul .h5
h5_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.h5"
with h5py.File(h5_file, "r") as h5:
    domain_data = np.array(h5["/data2"], dtype=np.int32)

# creează meshtags pentru domenii
cell_indices = np.arange(len(triangle_domains), dtype=np.int32)
cell_tags = meshtags(mesh, mesh.topology.dim, cell_indices, domain_data)

# verifică domeniile
print("domenii unice (din .h5):", np.unique(cell_tags.values))
#################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile
from mpi4py import MPI

# citește mesh-ul din fișierul .xdmf
xdmf_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.xdmf"
with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# extrage coordonatele nodurilor și conectivitatea triunghiurilor
points = mesh.geometry.x
cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)

# culori pentru fiecare domeniu (presupunem că ai cell_tags definit)
colors = cell_tags.values

# desenăm mesh-ul colorat pe domenii
plt.tripcolor(
    points[:, 0], points[:, 1], cells, facecolors=colors,
    cmap="tab20", edgecolors="k", linewidth=0.2
)

plt.gca().set_aspect("equal")
plt.colorbar(label="Domenii")
plt.title("Mesh colorat pe domenii")
plt.show()

#################################################################################################
# definirea spațiului funcțional
cell = "triangle"
element = basix.ufl.element("Lagrange", cell, 1)
vv = functionspace(mesh, element)
uu = TrialFunction(vv)
v = TestFunction(vv)

# definirea măsurilor de integrare
dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)

# definirea parametrilor variabili
mu0 = 4.0 * np.pi * 1e-7
mu_r_fe = 4000
mu_fe = mu_r_fe * mu0
j0 = 3.4e6

materials = {
    2: (mu_fe, 0.0, "iron"),
    1: (mu0, 0.0, "air"),
    3: (mu0, j0, "copper"),
}

mu = Function(vv)
j = Function(vv)

# atribuie valori pentru mu și j pe noduri
mu_values = {1: mu_fe, 2: mu0, 3: mu0}
j_values = {1: 0.0, 2: 0.0, 3: j0}
mu.x.array[:] = map_cell_tags_to_nodes(mesh, cell_tags, mu_values)
j.x.array[:] = map_cell_tags_to_nodes(mesh, cell_tags, j_values)

# forma variațională
a = inner((1 / mu) * grad(uu), grad(v)) * dx(1) + \
    inner((1 / mu) * grad(uu), grad(v)) * dx(2) + \
    inner((1 / mu) * grad(uu), grad(v)) * dx(3)
l = inner(j, v) * dx(1) + inner(j, v) * dx(2) + inner(j, v) * dx(3)

# compilează formele
a_form = form(a)
l_form = form(l)

# condiții la frontieră (dirichlet)
tdim = mesh.topology.dim
fdim = tdim - 1
dirichlet_facets = locate_entities_boundary(mesh, fdim, lambda x: ~np.isclose(x[1], 0.0))
dofs = locate_dofs_geometrical(vv, lambda x: ~np.isclose(x[1], 0.0))
bc = dirichletbc(PETSc.ScalarType(0), dofs, vv)

# asamblarea matricii și vectorului
a_matrix = assemble_matrix(a_form, bcs=[bc])
a_matrix.assemble()
b_vector = assemble_vector(l_form)
apply_lifting(b_vector, [a_form], [[bc]])
b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b_vector, [bc])

# rezolvă sistemul liniar
a_z = Function(vv)
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(a_matrix)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.solve(b_vector, a_z.x.petsc_vec)

# vizualizare
az_grid = pyvista.UnstructuredGrid(*vtk_mesh(vv))
az_grid.point_data["a_z"] = a_z.x.array.real
az_grid.set_active_scalars("a_z")
plotter = pyvista.Plotter()
plotter.add_mesh(az_grid, cmap="reds", show_edges=True, scalar_bar_args={"title": "a_z"})
plotter.view_xy()
plotter.show()


###################################################################################################################
# # # # # ######################################
# # # # # #######################################################################################################################
# # # # # # As we have computed the magnetic potential, we can now compute the magnetic field, by setting B=curl(A_z). Note that as we have chosen a function space of first order piecewise linear function to describe our potential, the curl of a function in this space is a discontinous zeroth order function (a function of cell-wise constants). We use dolfinx.fem.Expression to interpolate the curl into W.
# # # # # #######################################################################################################################
# with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.xdmf", "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")
#     ct = xdmf.read_meshtags(domain, name="Grid")   # <- foarte important#

from dolfinx.fem import functionspace, Function, Expression
import ufl
from ufl import as_vector
import pyvista
from dolfinx.plot import vtk_mesh
import basix
from dolfinx.fem.petsc import LinearProblem
# --- 1. Spațiu de funcții vectorial DG0 (discontinuu pe celule) ---
dim = 2  # dimensiune spațiu fizic
with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   # <- foarte important#

W = functionspace(domain, ("Discontinuous Lagrange", 0, (dim,)))

B = Function(W)
# --- 2. Calculează B = curl(A_z) prin finite differences pe celule (evită UFL ambiguu) ---
# --- 2. Calculează B = curl(A_z) cu L2 projection via LinearProblem (merge în versiuni vechi) ---

# Expression UFL pentru curl(A_z) în 2D: B = (∂A_z/∂y, -∂A_z/∂x)
B_expr = as_vector([a_z.dx(1), -a_z.dx(0)])

# Trial și test pentru W (vector DG0)
u_proj = TrialFunction(W)
v_proj = TestFunction(W)

# Forme pentru L2 projection: ∫ u_proj * v_proj dx = ∫ B_expr * v_proj dx
a_proj = inner(u_proj, v_proj) * dx  # bilinear (masă)
L_proj = inner(B_expr, v_proj) * dx  # linear (right-hand side cu curl)

# Rezolvă projection (fără BC-uri, DG0 nu are; solver local per celulă)
problem_proj = LinearProblem(a_proj, L_proj, u=B)  # u=B se umple cu soluția
problem_proj.solve()


# # ###################
# # --- 3. Pregătirea pentru vizualizare PyVista ---
from dolfinx.plot import vtk_mesh  # asigură-te că e importat sus

B_grid = pyvista.UnstructuredGrid(*vtk_mesh(domain))  # folosește domain, nu W (suportă DG0 cell_data)
B_mag = np.linalg.norm(B.x.array.reshape(-1, dim), axis=1)  # |B| pe celule (shape 927)
B_grid.cell_data["|B|"] = B_mag  # atașează pe celule, nu pe puncte
B_grid.set_active_scalars("|B|")

####### ################# --- 4. Vizualizare ---
# plotter = pyvista.Plotter()
# plotter.add_mesh(
#     B_grid,
#     cmap="Reds",
#     show_edges=True,
#     scalar_bar_args={"title": "Magnetic field |B|", "vertical": True}
# )
# plotter.view_xy()
# plotter.add_title("Magnetic Field |B| from curl(A_z)", font_size=12)

# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     plotter.screenshot("B_magnitude.png", window_size=[800, 800])

# # # # # ========================================================
# # # # # NOU: Grafic cu liniile de câmp (streamlines pentru B) - Versiune cu quiver minimal
# # # # # ========================================================
# # #######
# # --- pregatire date (folosim vectors_cell = (Bx, By, 0)) ---
cell_centers = B_grid.cell_centers()
B_values = B.x.array.reshape(-1, dim)     # (Ncells, 2)
vectors_cell = np.column_stack([B_values[:, 0], B_values[:, 1], np.zeros(len(B_values))])
# magnitudine B per celula
B_mag = np.linalg.norm(B_values, axis=1)

# --- calculez un factor rezonabil in functie de extinderea geometriei ---
x_extent = B_grid.bounds[1] - B_grid.bounds[0]
y_extent = B_grid.bounds[3] - B_grid.bounds[2]
domain_size = max(x_extent, y_extent, 1.0)  # evit division by zero

max_B = np.max(B_mag)
# desired arrow length ca fractiune din dimensiunea domeniului (ajusteaza 0.03 dupa nevoie)
desired_arrow_length = 0.01 * domain_size
if max_B > 0:
    # folosim factor = lungimea dorita / magnitudinea maxima pentru a obtine sageti vizibile, dar finite
    factor = desired_arrow_length / max_B
else:
    factor = desired_arrow_length

# Variante:
# 1) Săgeți orientate și scalate proporțional cu B (dar control absolute via factor)
# quiver = cell_centers.glyph(orient="B", scale="B", factor=factor)

# 2) Săgeți normalizate (arată doar direcția) cu lungime fixă = desired_arrow_length
# Recomand aceasta dacă magnitudele B sunt foarte mari și nu vrei variație de scară.
B_norm = np.zeros_like(vectors_cell)
norms = np.linalg.norm(vectors_cell[:, :2], axis=1, keepdims=True)
norms[norms == 0] = 1.0
B_norm[:, 0:2] = vectors_cell[:, 0:2] / norms
B_norm[:, 2] = 0.0

# pun vectorii normalizați in cell_centers ca field nou
cell_centers.cell_data.clear()  # curat date vechi
cell_centers.cell_data["Bnorm"] = B_norm

# glyph pe Bnorm, fara scalare automata, factor = desired_arrow_length
quiver = cell_centers.glyph(orient="Bnorm", scale=False, factor=desired_arrow_length)

# --- linii scurte pentru a sugera directia curentului (optional) ---
all_lines = []
n_lines = 20
if len(cell_centers.points) >= n_lines:
    selected_centers_idx = np.random.choice(len(cell_centers.points), n_lines, replace=False)
else:
    selected_centers_idx = np.arange(len(cell_centers.points))

for idx in selected_centers_idx:
    center = cell_centers.points[idx]
    b_local = vectors_cell[idx, :2]
    b_norm = np.linalg.norm(b_local)
    if b_norm > 1e-9:
        direction = b_local / b_norm
        # lungime liniuta dependenta de dimensiunea domeniului (nu de magnitudinea B)
        line_length = 0.02 * domain_size
        line_end = center + np.array([direction[0], direction[1], 0.0]) * line_length
        line = pyvista.Line(center, line_end)
        all_lines.append(line)

if all_lines:
    lines_combined = all_lines[0]
    for line in all_lines[1:]:
        lines_combined = lines_combined.merge(line)
else:
    lines_combined = pyvista.PolyData()  # gol

# --- Plot 2D (parallel projection, vedere "de sus") ---
plotter_lines = pyvista.Plotter(window_size=[1200, 900])
# arata celulele colorate dupa |B|
# colormaps = ["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow", "gray", "Greys", "binary", "coolwarm", "RdBu", "seismic", "turbo"]
plotter_lines.add_mesh(B_grid, cmap="magma", show_edges=True, opacity=0.5,
                       lighting=False,  # aspect 2D plat
                       scalar_bar_args={"title": "|B| (T)"})
# quiver rosu (sagetile normalizate, lungime fixa)
plotter_lines.add_mesh(quiver, color="red", opacity=0.9, show_scalar_bar=False)

# linii directionale subtiri (culoare galbena)
plotter_lines.add_mesh(lines_combined, color="yellow", line_width=2, opacity=1.0)

# setare camera 2D (view from top, parallel)
plotter_lines.view_xy()  # asiguram vedere de sus
plotter_lines.camera.parallel_projection = True

# Evidențiază conturul fiecărui domeniu
for d in np.unique(domains):
    # extrage celulele domeniului d
    mask = domains == d
    cells_d = B_grid.extract_cells(np.where(mask)[0])
    plotter_lines.add_mesh(
        cells_d,
        color=None,          # interior transparent
        show_edges=True,     # afișează muchiile
        edge_color="black",  # culoare contur
        line_width=2,
        opacity=0            # interior complet transparent
    )



center_x = (B_grid.bounds[0] + B_grid.bounds[1]) / 2.0
center_y = (B_grid.bounds[2] + B_grid.bounds[3]) / 2.0
# pozitionam camera deasupra planului la z>0 (nu afecteaza paralel projection)
plotter_lines.camera.position = (center_x, center_y, 1.0 * domain_size)
plotter_lines.camera.focal_point = (center_x, center_y, 0.0)
plotter_lines.camera.viewup = (0, 1, 0)

plotter_lines.add_title("Magnetic Field Lines (normalized arrows, 2D view)", font_size=14)
plotter_lines.show_grid()  # optional

# Afisare
if not pyvista.OFF_SCREEN:
    plotter_lines.show()
else:
    plotter_lines.screenshot("B_field_quiver_lines_2D.png", window_size=[1200, 900])
