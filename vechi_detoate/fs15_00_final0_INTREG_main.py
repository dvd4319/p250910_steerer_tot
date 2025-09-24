from fs15_01_nods_edg_tri_plot import (f01_load_comsol_mphtxt, f02_build_adjacency, f03_find_domains, f04_convert_mphtxt_to_gmsh, f05_load_comsol_msh, f06_convert_msh_to_xdmf, f07_extract_triangles_to_xdmf, f08_inspect_xdmf_mesh, f09_read_xdmf_mesh)
from fs15_01_nods_edg_tri_plot import (p01_plot_mesh_mphtxt,  p02_plot_mesh_with_labels, p03_plot_domains_mphtxt, 
                                       p04_plot_cell_tags, p05_plot_domains_gmesh1, p06_visualize_xdmf_mesh, p07_plot_subdomains)
########################################################################################################
from fs15_plot_granite import plot_external_boundary, plot_dirichlet_neumann_boundaries, plot_subdomains_tris
from fs15_solve_problem import assign_material_properties, define_boundary_conditions
from fs15_solve_problem import   assign_materials, plot_materials_on_mesh, assign_materials_variant, plot_materials_on_mesh_variant
from vechi_detoate.fs15_constructie_muchii import  build_edge_to_triangles, find_interface_edges, group_edges_by_domain, plot_interface_edges, save_domain_node_lists
########################################################################################################
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
from mpi4py import MPI
# #############################################################
# ### ================= MPHTXT ============================ ###
# #############################################################
# mphtxt_file = "comsol2dfara_spire_1pe8_vechi1_491_492.mphtxt"
mphtxt_file = "comsol2dfara_spire_1pe8_vechi1_501.mphtxt"
nodes, tris, edgs, tri_domains = f01_load_comsol_mphtxt(mphtxt_file)
print("###############################################")
print("DATA COLLECTED FROM .mphtxt")
print("## ======================================= ##")
print("Nodes mphtxt:", nodes.shape)
print("Triangles mphtxt:", tris.shape)
print("Edges mphtxt:", edgs.shape)
print("## ======================================= ##")
print("###############################################")

adj = f02_build_adjacency(tris, edgs)
domains = f03_find_domains(tris, adj)
print("###############################################")
print("DATA COLLECTED FROM .mphtxt")
print("## ======================================= ##")
# print("adj mphtxt:", adj)
# print("domains mphtxt:", domains)
print("## ======================================= ##")
print("###############################################")

# p01_plot_mesh_mphtxt(nodes, tris, edgs, title="Fig. (1) Mesh: 2D section - format .mphtxt")
# p02_plot_mesh_with_labels(nodes, tris, edgs, title="Fig. (2) Mesh with numbered nodes and red lines indicating the domain boundaries")

# Example: define materials on domains manually
domain_materials = {
    0: "AIR 1",
    1: "AIR 2",
    2: "IRON 1",
    3: "COPPER",
    4: "IRON 2",
}

# Colors defined manually (in the same order as IDs)
domain_colors = {
    0: "#314B7A",  # Air 
    1: "#314B7A",  # Air → light blue
    2: "#646363",  # Iron → copper-brown
    3: "#B87333",  # Copper → grey
    4: "#646363",  # Iron → copper-brown
}

#  p03_plot_domains_mphtxt(nodes, tris, edgs, domains, domain_materials, domain_colors, title="Fig. (3) domains - format .mphtxt")
# ###########################

# # #############################################################
# # ### ================= mphtxt TO msh ===================== ###
# # #############################################################
# msh_file = "comsol2dfara_spire_1pe8_vechi1_491_492.msh" ### generate msh 
msh_file = "comsol2dfara_spire_1pe8_vechi1_501.msh" ### generate msh 
f04_convert_mphtxt_to_gmsh(mphtxt_file,msh_file) ## convert to msh and display 
nodes, triangles, edges = f05_load_comsol_msh(msh_file) ### load msh for inspection 
# p05_plot_domains_gmesh1(nodes, triangles, edges, title="Fig. (5) 2D section - from format .msh")

# #############################################################
# ### ================= msh TO xdmf ======================= ###
# #############################################################
xdmf_path = f06_convert_msh_to_xdmf(msh_file) ### convert msh to xdmf 
xdmf_file = f07_extract_triangles_to_xdmf(msh_file) ### extract only triangles 
# p04_plot_cell_tags(xdmf_file,title = "Fig. (4) Mesh in format .xdmf (for FEniCS)", cmap="Accent")
domain, V = f08_inspect_xdmf_mesh(xdmf_file )
domain = f09_read_xdmf_mesh(xdmf_file)
# domain = p06_visualize_xdmf_mesh(xdmf_file, title = "Fig. (6) Mesh in format .xdmf (for FEniCS)")

# #############################################################
with XDMFFile(MPI.COMM_WORLD, xdmf_file , "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
# plot_cell_tags(xdmf_file) # green plot, only triangles green 
print("tri_domains shape:", tri_domains.shape)
print("tris shape:", tris.shape)
# plot_subdomains_tris(nodes, tris, domains) # domains colored , without labels 
# ##########################################################################################
# # list of triangles per domain (the 5 domains) - NOT A PLOT 
unique_domains = np.unique(domains)
for d in unique_domains:
    tri_indices = np.where(domains == d)[0]   # which triangles have tag d
    print(f"\nDomain {d}: {len(tri_indices)} triangles")
    for i in tri_indices:
        print(f"  Triangle {i}: nodes {tris[i]}")

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


# material values (corrected)
mu0 = 4.0 * np.pi * 1e-7
mu_r_FE = 4000
mu_FE = mu_r_FE * mu0
J0 = 3.4e6
materials2 = {
    0: (mu0,            0.0, "vacuum"),
    1: (mu0,            0.0, "air"),
    2: (mu0 * mu_r_FE,  0.0, "iron"),
    3: (mu0,       J0,   "copper (coil)"),
    4: (mu0 * mu_r_FE,  0.0, "iron")
}

mu, J = assign_materials(domain, materials2, domains)
# plot_materials_on_mesh(nodes, tris, domains, materials2)
# # ######################################################################################
# # #######################################################################################################################
# #######################################################################################################################
# # define the weak problem
# #######################################################################################################################
# # #######################################################################################################################

# with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_927_triangles.xdmf", "r") as xdmf:
with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_501_triangles.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   # <- very important#

VV = functionspace(domain, ("CG", 2))  # continuous space
uu = TrialFunction(VV)
vv = TestFunction(VV)


mu_val = mu_FE
J_val = 3.4e6

a = (1/mu_val) * dot(grad(uu), grad(vv)) * dx
L = J_val * vv * dx

tdim = domain.topology.dim
facets = locate_entities_boundary(domain, tdim - 1, lambda x: np.full(x.shape[1], True))
dofs = locate_dofs_topological(VV, tdim - 1, facets)
bc = dirichletbc(default_scalar_type(0), dofs, VV)

A_z = Function(VV)
problem = LinearProblem(a, L, u=A_z, bcs=[bc])
problem.solve()




Az_grid = pyvista.UnstructuredGrid(*vtk_mesh(VV))
Az_grid.point_data["A_z"] = A_z.x.array
Az_grid.set_active_scalars("A_z")

plotter = pyvista.Plotter()
plotter.add_mesh(Az_grid, cmap="Reds", show_edges=True, scalar_bar_args={"title":"A_z"})
plotter.view_xy()
plotter.show()




# ######################################
# #######################################################################################################################
# # As we have computed the magnetic potential, we can now compute the magnetic field, by setting B=curl(A_z). Note that as we have chosen a function space of first order piecewise linear function to describe our potential, the curl of a function in this space is a discontinous zeroth order function (a function of cell-wise constants). We use dolfinx.fem.Expression to interpolate the curl into W.
# #######################################################################################################################
from dolfinx.fem import functionspace, Function, Expression
import ufl
from ufl import as_vector
import pyvista
from dolfinx.plot import vtk_mesh
import basix
# --- 1. Vector function space DG0 (discontinuous per cell) ---
dim = 2  # physical space dimension

# Create DG0 vector space directly with a tuple
W = functionspace(domain, ("Discontinuous Lagrange", 0, (dim,)))

B = Function(W)
# --- 2. Compute B = curl(A_z) by finite differences on cells (avoids ambiguous UFL) ---
# --- 2. Compute B = curl(A_z) with L2 projection via LinearProblem (works in older versions) ---

# UFL Expression for curl(A_z) in 2D: B = (∂A_z/∂y, -∂A_z/∂x)
B_expr = as_vector([A_z.dx(1), -A_z.dx(0)])

# Trial and test for W (vector DG0)
u_proj = TrialFunction(W)
v_proj = TestFunction(W)

# Forms for L2 projection: ∫ u_proj * v_proj dx = ∫ B_expr * v_proj dx
a_proj = inner(u_proj, v_proj) * dx  # bilinear (mass)
L_proj = inner(B_expr, v_proj) * dx  # linear (right-hand side with curl)

# Solve projection (no BCs, DG0 has none; local per cell solver)
problem_proj = LinearProblem(a_proj, L_proj, u=B)  # u=B is filled with the solution
problem_proj.solve()


# ###################
# # --- 3. Prepare for PyVista visualization ---
from dolfinx.plot import vtk_mesh  # make sure it's imported above

B_grid = pyvista.UnstructuredGrid(*vtk_mesh(domain))  # use domain, not W (supports DG0 cell_data)
B_mag = np.linalg.norm(B.x.array.reshape(-1, dim), axis=1)  # |B| per cell (shape 927)
B_grid.cell_data["|B|"] = B_mag  # attach to cells, not points
B_grid.set_active_scalars("|B|")

# --- 4. Visualization ---
plotter = pyvista.Plotter()
plotter.add_mesh(
    B_grid,
    cmap="Reds",
    show_edges=True,
    scalar_bar_args={"title": "Magnetic field |B|", "vertical": True}
)
plotter.view_xy()
plotter.add_title("Magnetic Field |B| from curl(A_z)", font_size=12)

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("B_magnitude.png", window_size=[800, 800])
