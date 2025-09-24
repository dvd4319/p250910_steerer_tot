from fs15_01_nods_edg_tri_plot import build_adjacency, load_comsol_mphtxt, plot_mesh
########################################################################################
from fs15_02_util import find_domains, convert_mphtxt_to_gmsh, load_comsol_msh, convert_msh_to_xdmf
from fs15_02_util import extract_triangles_to_xdmf, inspect_xdmf_mesh, read_xdmf_mesh
########################################################################################################
from fs15_plot import  plot_mesh_with_labels, plot_domains, plot_cell_tags, plot_comsol_mesh
from fs15_plot import visualize_xdmf_mesh, plot_subdomains
########################################################################################################
from fs15_subdomains_data_501 import subdomains, colors, labels
from fs15_plot_granite import plot_external_boundary, plot_dirichlet_neumann_boundaries, plot_subdomains_tris
from vechi_detoate.fs15_solve_problem import assign_material_properties, define_boundary_conditions
from vechi_detoate.fs15_solve_problem import   assign_materials, plot_materials_on_mesh, assign_materials_variant, plot_materials_on_mesh_variant
########################################################################################################
from dolfinx.io.gmshio import read_from_msh, model_to_mesh
from dolfinx.io import XDMFFile
from dolfinx import mesh, fem, plot
from dolfinx.mesh import locate_entities_boundary, compute_midpoints, meshtags
from dolfinx.fem import dirichletbc, Expression, Function, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, as_vector, dot, dx, grad, inner
import ufl 
from dolfinx import default_scalar_type
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import compute_midpoints, locate_entities_boundary
from dolfinx.plot import vtk_mesh

########################################################################################################
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista
##########################################################
rank = MPI.COMM_WORLD.rank
gdim = 2
model_rank = 0
mesh_comm = MPI.COMM_WORLD
MPI.COMM_WORLD.barrier()
comm = MPI.COMM_WORLD
#############################################################
### ================= MPHTXT ============================ ###
#############################################################
# mphtxt_file = "comsol2dfara_spire_1pe8_vechi1_491_492.mphtxt"
mphtxt_file = "comsol2dfara_spire_1pe8_vechi1.mphtxt"
nodes, tris, edgs, tri_domains = load_comsol_mphtxt(mphtxt_file)
adj = build_adjacency(tris, edgs)
domains = find_domains(tris, adj)
print("Nodes:", nodes.shape)
print("Triangles:", tris.shape)
print("Edges:", edgs.shape)
plot_mesh(nodes, tris, edgs, title="Fig. (1) Mesh: 1/4 2D section (x<0, y>0) - format .mphtxt")
#plot_mesh_with_labels(nodes, tris, edgs, title="Fig. (2) Mesh with numbered nodes and red lines indicating the domain boundaries")
#plot_domains(nodes, tris, edgs, domains, title="Fig. (3) Domenii identificate automat")

#############################################################
### ================= mphtxt IN msh ===================== ###
# #############################################################
# msh_file = "comsol2dfara_spire_1pe8_vechi1_491_492.msh" ### inventeaza msh 
# convert_mphtxt_to_gmsh(mphtxt_file,msh_file) ## tranforma in msh si afiseaza 
# # nodes, triangles, edges = load_comsol_msh(msh_file) ### incarca msh ca sa fie vazut 
# #plot_comsol_mesh(nodes, triangles, edges, title="Fig. (4) 1/4  2D section (x<0, y>0) - from format .msh")

# #############################################################
# ### ================= msh IN xdmf ======================= ###
# #############################################################
# xdmf_path = convert_msh_to_xdmf(msh_file) ### converteste msh in xdmf 
# xdmf_file = extract_triangles_to_xdmf(msh_file) ### extrage doar tiunghiurile 
# domain, V = inspect_xdmf_mesh(xdmf_file )
# # domain = read_xdmf_mesh(xdmf_file)
# # domain = visualize_xdmf_mesh(xdmf_file, title = "Fig. (5) Mesh in format .xdmf (pentru FEniCS)")

# #############################################################
# ### ================= SUB-DOMAINS ======================= ###
# #############################################################
# # plot_subdomains(nodes, tris, edgs, subdomains, colors, labels, figsize=(10,10))
# # external_boundary = plot_external_boundary(nodes, subdomains)
# # plot_dirichlet_neumann_boundaries(nodes, tris, tol=1e-12, title = "Fig. (9) Frontiere Dirichlet (roșu) și Neumann (verde)")
# #############################################################
# xdmf_file = "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf"
# with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf", "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")
# # plot_cell_tags(xdmf_file) # grafic verde, doar triunghiuri verzi 
# print("tri_domains shape:", tri_domains.shape)
# print("tris shape:", tris.shape)
# domains = find_domains(tris, adj)
# #plot_subdomains_tris(nodes, tris, domains) # domenii colorate , fara etichete 
# ##############################################################################
# # print("Number of triangles:", tris.shape[0])
# # print("Number of domain tags:", tri_domains.shape[0])
# ##########################################################################################
# # lista cu triunghiurile domeniilor (cele 5 domenii) - NU CRAFIC 

# # unique_domains = np.unique(domains)
# # for d in unique_domains:
# #     tri_indices = np.where(domains == d)[0]   # care triunghiuri au tag-ul d
# #     print(f"\nDomeniu {d}: {len(tri_indices)} triunghiuri")
# #     for i in tri_indices:
# #         print(f"  Triunghi {i}: noduri {tris[i]}")
# ####################################################################################
# mu_FE = 4000
# J0 = 4e6
# materials1 = {
#     0: (4*np.pi*1e-7, 0.0, "vacuum"),
#     1: (1e-5, 0.0,"air"),
#     2: (mu_FE, 0.0,"iron"),
#     3: (1.26e-6, J0, "copper (coil)"),
#     4: (mu_FE, 0.0, "iron")
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
# # We can now define the weak problem
# #######################################################################################################################
# # #######################################################################################################################

# # with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf", "r") as xdmf:
# #     domain = xdmf.read_mesh(name="Grid")
# #     ct = xdmf.read_meshtags(domain, name="Grid")   # <- foarte important#

# # VV = functionspace(domain, ("DG", 1))
# # tdim = domain.topology.dim
# # facets = locate_entities_boundary(domain, tdim - 1, lambda x: np.full(x.shape[1], True))
# # dofs = locate_dofs_topological(VV, tdim - 1, facets)
# # # # bc = dirichletbc(PETSc.ScalarType(0), dofs, V)
# # bc = dirichletbc(default_scalar_type(0), dofs, VV)


# # uu = TrialFunction(VV)
# # vv = TestFunction(VV)
# # # a = (1/mu) * dot(grad(uu), grad(vv)) * ufl.dx(domain=domain)
# # # L = J * vv * ufl.dx(domain=domain)
# # mu_val = 4000.0
# # J_val  = 4e6

# # a = (1/mu_val) * dot(grad(uu), grad(vv)) * dx
# # L = J_val * vv * dx



# # #######################################################################################################################
# # # Solve the linear problem
# # #######################################################################################################################
# # A_z = Function(VV)
# # problem = LinearProblem(a, L, u=A_z, bcs=[bc])
# # problem.solve()


# ##########################
# import numpy as np
# import pyvista
# from dolfinx import fem, mesh, plot
# from dolfinx.fem import Function, functionspace, dirichletbc
# from dolfinx.fem.petsc import LinearProblem
# from dolfinx.io import XDMFFile
# from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
# from ufl import TestFunction, TrialFunction, dot, dx, grad
# from mpi4py import MPI

# # Load tri_domains from the COMSOL file
# from fs15_util import load_comsol_mphtxt
# mphtxt_file = "comsol2dfara_spire_1pe8_vechi1_491_492.mphtxt"
# nodes, tris, edgs, tri_domains = load_comsol_mphtxt(mphtxt_file)
# print("tri_domains shape:", tri_domains.shape)
# unique, counts = np.unique(tri_domains, return_counts=True)
# print("Unique tri_domains:", dict(zip(unique, counts)))

# # Validate and correct tri_domains (map invalid tags 5, 6, 927 to 0)
# valid_tags = {0, 1, 2, 3, 4}
# tri_domains = np.where(np.isin(tri_domains, list(valid_tags)), tri_domains, 0)
# unique, counts = np.unique(tri_domains, return_counts=True)
# print("Corrected unique tri_domains:", dict(zip(unique, counts)))

# # Pad tri_domains to match number of triangles
# num_tris = tris.shape[0]  # 926
# if tri_domains.shape[0] != num_tris:
#     print("Warning: tri_domains length != number of triangles. Padding with 0.")
#     domains = np.zeros(num_tris, dtype=np.int32)
#     domains[:tri_domains.shape[0]] = tri_domains
# else:
#     domains = tri_domains

# # Load the mesh from XDMF
# xdmf_file = "comsol2dfara_spire_1pe8_vechi1_491_492_triangles.xdmf"
# with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")

# # Create new meshtags using corrected tri_domains
# num_cells = domain.topology.index_map(domain.topology.dim).size_local
# print(f"Number of cells in mesh: {num_cells}")
# ct = mesh.meshtags(domain, domain.topology.dim, np.arange(num_cells, dtype=np.int32), domains)
# unique, counts = np.unique(ct.values, return_counts=True)
# print(f"New cell tags shape: {ct.values.shape}")
# print(f"New unique cell tags: {dict(zip(unique, counts))}")

# # Define function space (DG 1 for A_z computation)
# VV_dg = functionspace(domain, ("DG", 1))
# tdim = domain.topology.dim

# # Define boundary condition (Dirichlet A_z = 0 on all boundaries)
# facets = mesh.locate_entities_boundary(domain, tdim - 1, lambda x: np.full(x.shape[1], True))
# dofs = fem.locate_dofs_topological(VV_dg, tdim - 1, facets)
# bc = dirichletbc(0.0, dofs, VV_dg)

# # Material properties
# mu0 = 4.0 * np.pi * 1e-7
# materials = {
#     0: (mu0, 0.0, "vacuum"),
#     1: (mu0, 0.0, "air"),
#     2: (mu0 * 100.0, 0.0, "iron"),
#     3: (mu0, 4e6, "copper (coil)"),
#     4: (mu0 * 100.0, 0.0, "iron")
# }

# # Assign material properties using new cell tags
# mu = Function(VV_dg)
# J = Function(VV_dg)
# cells = ct.indices
# tags = ct.values
# for tag, (mu_val, J_val, name) in materials.items():
#     if np.any(tags == tag):
#         mu.x.array[cells[tags == tag]] = mu_val
#         J.x.array[cells[tags == tag]] = J_val
#         print(f"Assigned material {name} (tag {tag}) to {np.sum(tags == tag)} cells")
#     else:
#         print(f"Warning: No cells found with tag {tag}")

# # Define weak form
# uu = TrialFunction(VV_dg)
# vv = TestFunction(VV_dg)
# a = (1 / mu) * dot(grad(uu), grad(vv)) * dx(domain=domain)
# L = J * vv * dx(domain=domain)

# # Solve the linear problem
# A_z = Function(VV_dg)
# problem = LinearProblem(a, L, u=A_z, bcs=[bc])
# problem.solve()

# # Interpolate A_z to CG 1 for XDMF output
# VV_cg = functionspace(domain, ("CG", 1))
# A_z_cg = Function(VV_cg)
# A_z_cg.interpolate(A_z)

# # Save A_z to XDMF for inspection
# with XDMFFile(MPI.COMM_WORLD, "Az_output.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(A_z_cg)

# # Visualization with PyVista
# points = domain.geometry.x
# points_3d = np.hstack([points, np.zeros((points.shape[0], 1))])
# points_3d = np.ascontiguousarray(points_3d, dtype=np.float64)
# points_3d.setflags(write=False)
# print(f"points_3d shape: {points_3d.shape}, dtype: {points_3d.dtype}, C-contiguous: {points_3d.flags['C_CONTIGUOUS']}")

# # Compute cell assignments for points
# bb_tree = bb_tree(domain, domain.topology.dim)
# cell_candidates = compute_collisions_points(bb_tree, points_3d)
# colliding_cells = compute_colliding_cells(domain, cell_candidates, points_3d)
# cells = np.array([c[0] if len(c) > 0 else 0 for c in colliding_cells], dtype=np.int32)

# # Evaluate A_z at points
# Az_values = np.zeros((points.shape[0], 1), dtype=np.float64)
# A_z.eval(points_3d, cells, Az_values)

# # Check for invalid values
# if np.any(np.isnan(Az_values)) or np.any(np.isinf(Az_values)):
#     print("Warning: Az_values contains NaN or Inf values")
# else:
#     print(f"A_z range: min={np.min(Az_values):.4e}, max={np.max(Az_values):.4e}")

# # Create PyVista grid
# Az_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain))
# Az_grid.point_data["A_z"] = Az_values.flatten()
# Az_grid.set_active_scalars("A_z")

# # Plot
# plotter = pyvista.Plotter()
# plotter.add_mesh(Az_grid, cmap="Reds", show_edges=True, scalar_bar_args={"title": "A_z", "vertical": True})
# plotter.view_xy()
# plotter.add_title("Magnetic Potential A_z", font_size=12)

# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     plotter.screenshot("Az_with_labels.png", window_size=[800, 800])