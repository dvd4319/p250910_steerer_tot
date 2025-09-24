from fs15_01_nods_edg_tri_plot import (f01_load_comsol_mphtxt, f02_build_adjacency, f03_find_domains, f04_convert_mphtxt_to_gmsh, f05_load_comsol_msh, f06_convert_msh_to_xdmf, f07_extract_triangles_to_xdmf, f08_inspect_xdmf_mesh, f09_read_xdmf_mesh)
from fs15_01_nods_edg_tri_plot import (p01_plot_mesh_mphtxt,  p02_plot_mesh_with_labels, p03_plot_domains_mphtxt, 
                                       p04_plot_cell_tags, p05_plot_domains_gmesh1, p06_visualize_xdmf_mesh, p07_plot_subdomains)
########################################################################################################
#from fs15_plot import visualize_xdmf_mesh, plot_subdomains
########################################################################################################
from fs15_plot_granite import plot_external_boundary, plot_dirichlet_neumann_boundaries, plot_subdomains_tris
from vechi_detoate.fs15_solve_problem import assign_material_properties, define_boundary_conditions
from vechi_detoate.fs15_solve_problem import   assign_materials, plot_materials_on_mesh, assign_materials_variant, plot_materials_on_mesh_variant
from vechi_detoate.fs15_constructie_muchii import  build_edge_to_triangles, find_interface_edges, group_edges_by_domain, plot_interface_edges, save_domain_node_lists
########################################################################################################
from dolfinx.io.gmshio import read_from_msh, model_to_mesh
from dolfinx.io import XDMFFile
from dolfinx import mesh, fem, plot
from dolfinx.mesh import locate_entities_boundary, compute_midpoints, meshtags
from dolfinx.fem import dirichletbc, Expression, Function, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, as_vector, dot, dx, grad, inner
from dolfinx import default_scalar_type
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import compute_midpoints, locate_entities_boundary
from dolfinx.plot import vtk_mesh
########################################################################################################
########################################################################################################
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista
import ufl 
##########################################################
rank = MPI.COMM_WORLD.rank
gdim = 2
model_rank = 0
mesh_comm = MPI.COMM_WORLD
MPI.COMM_WORLD.barrier()
comm = MPI.COMM_WORLD
# #############################################################
# ### ================= MPHTXT ============================ ###
# #############################################################
mphtxt_file = "comsol2dfara_spire_toata_vechi1_1862.mphtxt"
nodes, tris, edgs, tri_domains = f01_load_comsol_mphtxt(mphtxt_file)
print("###############################################")
print("DATELE SUNT CULESTE DIN .mphtxt")
print("## ======================================= ##")
print("Nodes mphtxt:", nodes.shape)
print("Triangles mphtxt:", tris.shape)
print("Edges mphtxt:", edgs.shape)
print("## ======================================= ##")
print("###############################################")

adj = f02_build_adjacency(tris, edgs)
domains = f03_find_domains(tris, adj)
print("###############################################")
print("DATELE SUNT CULESTE DIN .mphtxt")
print("## ======================================= ##")
# print("adj mphtxt:", adj)
# print("domains mphtxt:", domains)
print("## ======================================= ##")
print("###############################################")

p01_plot_mesh_mphtxt(nodes, tris, edgs, title="Fig. (1) Mesh: 2D section - format .mphtxt")
p02_plot_mesh_with_labels(nodes, tris, edgs, title="Fig. (2) Mesh with numbered nodes and red lines indicating the domain boundaries")


# Exemplu: definim manual materialele pe domenii
domain_materials = {
    0: "IRON 1",
    1: "AIR",
    2: "COIL_1 (PLUS)",
    3: "IRON_2",
    4: "COIL_2 (PLUS)",
    5: "COIL_1 (MINUS)",
    6: "COIL_2 (MINUS)",
    # etc... în funcție de ce ai tu în COMSOL
}

# Culori definite manual (în aceeași ordine ca ID-urile)
domain_colors = {
    0: "#808080",  # Steel → gri
    1: "#ADD8E6",  # Air → albastru deschis
    2: "#B87333",  # Copper → maro cupru
    3: "#808080",  # Steel → gri
    4: "#B87333",  # Copper → maro cupru
    5: "#B87333",  # Copper → maro cupru
    6: "#B87333",  # Copper → maro cupru
}
p03_plot_domains_mphtxt(nodes, tris, edgs, domains, domain_materials, domain_colors, title="Fig. (3) domains - format .mphtxt")
# ###########################

# # #############################################################
# # ### ================= mphtxt IN msh ===================== ###
# # #############################################################
msh_file = "comsol2dfara_spire_toata_vechi1_1862.msh" ### inventeaza msh 
f04_convert_mphtxt_to_gmsh(mphtxt_file,msh_file) ## tranforma in msh si afiseaza 
nodes, triangles, edges = f05_load_comsol_msh(msh_file) ### incarca msh ca sa fie vazut 
p05_plot_domains_gmesh1(nodes, triangles, edges, title="Fig. (5) 2D section - from format .msh")

# #############################################################
# ### ================= msh IN xdmf ======================= ###
# #############################################################
xdmf_path = f06_convert_msh_to_xdmf(msh_file) ### converteste msh in xdmf 
xdmf_file = f07_extract_triangles_to_xdmf(msh_file) ### extrage doar tiunghiurile 
p04_plot_cell_tags(xdmf_file,title = "Fig. (4) Mesh in format .xdmf (pentru FEniCS)", cmap="Accent")
domain, V = f08_inspect_xdmf_mesh(xdmf_file )
domain = f09_read_xdmf_mesh(xdmf_file)
domain = p06_visualize_xdmf_mesh(xdmf_file, title = "Fig. (6) Mesh in format .xdmf (pentru FEniCS)")

# ############################################################
# ## ================= SUB-DOMAINS ======================= ###
# ############################################################
#plot_subdomains(nodes, tris, edgs, subdomains, colors, labels, figsize=(10,10))
# # external_boundary = plot_external_boundary(nodes, subdomains)
# # plot_dirichlet_neumann_boundaries(nodes, tris, tol=1e-12, title = "Fig. (9) Frontiere Dirichlet (roșu) și Neumann (verde)")
# #############################################################
# with XDMFFile(MPI.COMM_WORLD, xdmf_file , "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")
# # plot_cell_tags(xdmf_file) # grafic verde, doar triunghiuri verzi 
# print("tri_domains shape:", tri_domains.shape)
# print("tris shape:", tris.shape)
# domains = find_domains(tris, adj)
# plot_subdomains_tris(nodes, tris, domains) # domenii colorate , fara etichete 
# ##############################################################################
# print("Number of triangles:", tris.shape[0])
# print("Number of domain tags:", tri_domains.shape[0])
# ##########################################################################################
# # lista cu triunghiurile domeniilor (cele 5 domenii) - NU CRAFIC 

# # unique_domains = np.unique(domains)
# # for d in unique_domains:
# #     tri_indices = np.where(domains == d)[0]   # care triunghiuri au tag-ul d
# #     print(f"\nDomeniu {d}: {len(tri_indices)} triunghiuri")
# #     for i in tri_indices:
# #         print(f"  Triunghi {i}: noduri {tris[i]}")
# ####################################################################################
# mu_r_FE = 4000
# mu_FE = mu_r_FE * 4*np.pi*1e-7
# J0 = 3.4e6
# mu_0 = 4*np.pi*1e-7 
# materials1 = {
#     0: (mu_FE, 0.0, "Iron 1"),
#     1: (mu_0, 0.0,"Air"),
#     2: (mu_0, J0,"Coil 1 plus"),
#     3: (mu_FE, 00, "Iron 2"),
#     4: (mu_0, J0, "Coil 2 plus "),
#     5: (mu_0, -J0, "Coil 1 minus "),
#     6: (mu_0, -J0, "Coil 2 minus")
# }


# # valori materiale (corectate)
# mu0 = 4.0 * np.pi * 1e-7
# materials2 = {
#     0: (mu0,          0.0, "vacuum"),
#     1: (mu0,          0.0, "air"),
#     2: (mu0 * 100.0,  0.0, "iron"),
#     3: (mu0,       4e6,   "copper (coil)"),
#     4: (mu0 * 100.0,  0.0, "iron")
# }

# mu, J = assign_materials(domain, materials1, domains)
# plot_materials_on_mesh(nodes, tris, domains, materials1)
# # ######################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
# from dolfinx import default_scalar_type
# from dolfinx.fem import Function, functionspace

# # verificare lungimi (dacă e nevoie completează)
# if domains.shape[0] != tris.shape[0]:
#     print("Warning: domains length != #triangles. Padding missing entries with -1.")
#     tmp = np.full(tris.shape[0], -1, dtype=int)
#     tmp[:domains.shape[0]] = domains
#     domains = tmp


# # assign_materials_variant(materials2,domains)
# # plot_materials_on_mesh_variant(nodes, tris, domains, materials2)
# # #######################################################################################################################
# #######################################################################################################################
# # define the weak problem
# #######################################################################################################################
# # #######################################################################################################################

# # with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_927_triangles.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_toata_vechi1_1862_triangles.xdmf", "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")
#     ct = xdmf.read_meshtags(domain, name="Grid")   # <- foarte important#

# VV = functionspace(domain, ("CG", 2))  # spațiu continuu
# uu = TrialFunction(VV)
# vv = TestFunction(VV)


# mu_val = mu_FE
# J_val = 4e6

# a = (1/mu_val) * dot(grad(uu), grad(vv)) * dx
# L = J_val * vv * dx

# tdim = domain.topology.dim
# facets = locate_entities_boundary(domain, tdim - 1, lambda x: np.full(x.shape[1], True))
# dofs = locate_dofs_topological(VV, tdim - 1, facets)
# bc = dirichletbc(default_scalar_type(0), dofs, VV)

# A_z = Function(VV)
# problem = LinearProblem(a, L, u=A_z, bcs=[bc])
# problem.solve()

# Az_grid = pyvista.UnstructuredGrid(*vtk_mesh(VV))
# Az_grid.point_data["A_z"] = A_z.x.array
# Az_grid.set_active_scalars("A_z")

# plotter = pyvista.Plotter()
# plotter.add_mesh(Az_grid, cmap="Reds", show_edges=True, scalar_bar_args={"title":"A_z"})
# plotter.view_xy()
# plotter.show()




# ######################################
# #######################################################################################################################
# # As we have computed the magnetic potential, we can now compute the magnetic field, by setting B=curl(A_z). Note that as we have chosen a function space of first order piecewise linear function to describe our potential, the curl of a function in this space is a discontinous zeroth order function (a function of cell-wise constants). We use dolfinx.fem.Expression to interpolate the curl into W.
# #######################################################################################################################
# from dolfinx.fem import functionspace, Function, Expression
# import ufl
# from ufl import as_vector
# import pyvista
# from dolfinx.plot import vtk_mesh

# # --- 1. Spațiu de funcții vectorial DG0 (discontinuu pe celule) ---

# from dolfinx.fem import Function, functionspace
# from ufl import as_vector
# import basix
# # --- 1. Spațiu de funcții vectorial DG0 (discontinuu pe celule) ---
# dim = 2  # dimensiune spațiu fizic

# # Creează spațiul DG0 vectorial direct cu tuple
# W = functionspace(domain, ("Discontinuous Lagrange", 0, (dim,)))

# B = Function(W)
# # --- 2. Calculează B = curl(A_z) prin finite differences pe celule (evită UFL ambiguu) ---
# # --- 2. Calculează B = curl(A_z) cu L2 projection via LinearProblem (merge în versiuni vechi) ---

# # Expression UFL pentru curl(A_z) în 2D: B = (∂A_z/∂y, -∂A_z/∂x)
# B_expr = as_vector([A_z.dx(1), -A_z.dx(0)])

# # Trial și test pentru W (vector DG0)
# u_proj = TrialFunction(W)
# v_proj = TestFunction(W)

# # Forme pentru L2 projection: ∫ u_proj * v_proj dx = ∫ B_expr * v_proj dx
# a_proj = inner(u_proj, v_proj) * dx  # bilinear (masă)
# L_proj = inner(B_expr, v_proj) * dx  # linear (right-hand side cu curl)

# # Rezolvă projection (fără BC-uri, DG0 nu are; solver local per celulă)
# problem_proj = LinearProblem(a_proj, L_proj, u=B)  # u=B se umple cu soluția
# problem_proj.solve()


# ###################
# # --- 3. Pregătirea pentru vizualizare PyVista ---
# from dolfinx.plot import vtk_mesh  # asigură-te că e importat sus

# B_grid = pyvista.UnstructuredGrid(*vtk_mesh(domain))  # folosește domain, nu W (suportă DG0 cell_data)
# B_mag = np.linalg.norm(B.x.array.reshape(-1, dim), axis=1)  # |B| pe celule (shape 927)
# B_grid.cell_data["|B|"] = B_mag  # atașează pe celule, nu pe puncte
# B_grid.set_active_scalars("|B|")

# # --- 4. Vizualizare ---
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

