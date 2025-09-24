# # definirea spațiului funcțional
# cell = "triangle"
# element = basix.ufl.element("Lagrange", cell, 1)
# vv = functionspace(mesh, element)
# uu = TrialFunction(vv)
# v = TestFunction(vv)

# # from dolfinx.fem import Function, FunctionSpace

# # vv = FunctionSpace(mesh, ("CG", 1))  # Lagrange 1
# # mu = Function(vv)
# # j = Function(vv)



# # definirea măsurilor de integrare
# dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)

# # definirea parametrilor variabili
# mu0 = 4.0 * np.pi * 1e-7
# mu_r_fe = 4000
# mu_fe = mu_r_fe * mu0
# j0 = 3.4e6

# materials = {
#     1: (mu_fe, 0.0, "iron"),
#     2: (mu0, 0.0, "air"),
#     3: (mu0, j0, "copper"),
# }

# mu = Function(vv)
# j = Function(vv)

# # atribuie valori pentru mu și j pe noduri
# mu_values = {1: mu_fe, 2: mu0, 3: mu0}
# j_values = {1: 0.0, 2: 0.0, 3: j0}
# mu.x.array[:] = map_cell_tags_to_nodes(mesh, cell_tags, mu_values)
# j.x.array[:] = map_cell_tags_to_nodes(mesh, cell_tags, j_values)

# # Atribuie valoare fiecărui nod în funcție de domeniul celulei
# for cell_index in range(mesh.topology.index_map(mesh.topology.dim).size_local):
#     dom = cell_tags.values[cell_index]
#     for node in mesh.topology.connectivity(mesh.topology.dim, 0).links(cell_index):
#         mu.x.array[node] = mu_values[dom]
#         j.x.array[node] = j_values[dom]

# print("Domenii unice:", np.unique(cell_tags.values))
# print("Valorile j pe noduri:", np.unique(j.x.array))

# # forma variațională
# # a = inner((1 / mu) * grad(uu), grad(v)) * dx(1) + \
# #     inner((1 / mu) * grad(uu), grad(v)) * dx(2) + \
# #     inner((1 / mu) * grad(uu), grad(v)) * dx(3)
# # l = inner(j, v) * dx(1) + inner(j, v) * dx(2) + inner(j, v) * dx(3)

# dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)
# a = 0
# l = 0
# for dom_id in np.unique(cell_tags.values):
#     a += inner((1 / mu) * grad(uu), grad(v)) * dx(dom_id)
#     l += inner(j, v) * dx(dom_id)


# # compilează formele
# a_form = form(a)
# l_form = form(l)

# # condiții la frontieră (dirichlet)
# tdim = mesh.topology.dim
# fdim = tdim - 1
# dirichlet_facets = locate_entities_boundary(mesh, fdim, lambda x: ~np.isclose(x[1], 0.0))
# dofs = locate_dofs_geometrical(vv, lambda x: ~np.isclose(x[1], 0.0))
# bc = dirichletbc(PETSc.ScalarType(0), dofs, vv)

# node_j = map_cell_tags_to_nodes(mesh, cell_tags, j_values)
# print("Valori j noduri:", np.unique(node_j))

# # asamblarea matricii și vectorului
# a_matrix = assemble_matrix(a_form, bcs=[bc])
# a_matrix.assemble()
# b_vector = assemble_vector(l_form)
# apply_lifting(b_vector, [a_form], [[bc]])
# b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# set_bc(b_vector, [bc])

# # rezolvă sistemul liniar
# a_z = Function(vv)
# solver = PETSc.KSP().create(MPI.COMM_WORLD)
# solver.setOperators(a_matrix)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)
# solver.solve(b_vector, a_z.x.petsc_vec)

# # # vizualizare
# # az_grid = pyvista.UnstructuredGrid(*vtk_mesh(vv))
# # az_grid.point_data["a_z"] = a_z.x.array.real
# # az_grid.set_active_scalars("a_z")
# # plotter = pyvista.Plotter()
# # plotter.add_mesh(az_grid, cmap="reds", show_edges=True, scalar_bar_args={"title": "a_z"})
# # plotter.view_xy()
# # plotter.show()


# import pyvista as pv
# import numpy as np

# # --- presupunem că 'mesh', 'cell_tags', și 'a_z' sunt deja calculate ---

# # Creează PyVista grid pentru fundal (domenii)
# grid_domains = pyvista.UnstructuredGrid(*vtk_mesh(vv))
# grid_domains.cell_data["domain"] = cell_tags.values

# # Creează PyVista grid pentru rezultatul FEM
# grid_results = pv.UnstructuredGrid(*vtk_mesh(vv))
# grid_results.point_data["a_z"] = a_z.x.array.real
# grid_results.set_active_scalars("a_z")

# # Plot combinat
# plotter = pv.Plotter()
# # fundal: domenii colorate
# plotter.add_mesh(grid_domains, show_edges=True, scalars="domain", cmap="tab20", opacity=0.5)
# # suprapunere: A_z
# plotter.add_mesh(grid_results, show_edges=False, scalars="a_z", cmap="Reds", opacity=1.0)

# plotter.add_axes()                           # adaugă axele X/Y
# plotter.add_text("Distribuție A_z peste mesh", position="upper_edge", font_size=14, color="black")
# plotter.view_xy()
# plotter.show()


# ##############################################################################################

# with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
#     domain_2 = xdmf.read_mesh(name="Grid")
#     ct = xdmf.read_meshtags(domain_2, name="Grid")   # <- foarte important#
# # # # # valori materiale (corectate)
# # ==================== 1. Definire materiale ====================
# mu0 = 4.0 * np.pi * 1e-7
# mu_r_FE = 4000
# J0 = 3.4e6

# # Dicționar materiale pe domenii (conform tri_domains_corect / tri_domains_xdmf)
# materials_dict_2 = {
#     1: (mu0 * mu_r_FE, J0, "IRON"),
#     2: (mu0,          0.0, "AIR"),
#     3: (mu0,          0.0, "COPPER"),
# }

# # ==================== 2. Mapare materiale pe mesh ====================
# # Folosim tri_domains_xdmf ca vector de domenii pe triunghiuri
# mu, J = assign_materials(domain_2, materials_dict_2, tri_domains_xdmf_1)


# # ==================== 3. Definire spațiu funcții FEM ====================
# VV = functionspace(domain_2, ("CG", 1))  # spațiu continuu
# uu = TrialFunction(VV)
# vv = TestFunction(VV)

# # Dacă mu și J sunt scalare, folosește direct valorile
# # Dacă sunt array-uri distribuite pe noduri/triunghiuri, trebuie să creezi Function
# a = inner((1 / mu) * grad(uu), grad(vv)) * dx
# L = inner(J, vv) * dx

# # ==================== 4. Condiții Dirichlet ====================
# tdim = domain_2.topology.dim
# fdim = tdim - 1
# # exemplu: fixăm A_z = 0 pe frontiera y != 0
# dirichlet_facets = locate_entities_boundary(domain_2, fdim, lambda x: ~np.isclose(x[1], 0.0))
# dofs = locate_dofs_topological(VV, fdim, dirichlet_facets)
# bc = dirichletbc(default_scalar_type(0), dofs, VV)

# # ==================== 5. Rezolvare ====================
# A_z = Function(VV)
# problem = LinearProblem(a, L, u=A_z, bcs=[bc])
# problem.solve()

# # ==================== 6. Vizualizare PyVista ====================
# Az_grid = pyvista.UnstructuredGrid(*vtk_mesh(VV))
# Az_grid.point_data["A_z"] = A_z.x.array
# plotter = pyvista.Plotter()
# plotter.add_mesh(Az_grid, cmap="Reds", show_edges=True, scalar_bar_args={"title": "A_z"})
# plotter.view_xy()
# plotter.show()



# ##############################################################################################
# ##############################################################################################
# ##############################################################################################

# # # Asigură-te că triangle_domains este un array NumPy
# triangle_domains_xdmf = np.asarray(triangle_domains_xdmf, dtype=np.int32)

# # # Verifică dacă lungimile coincid
# if len(triangle_cells_xdmf) != len(triangle_domains_xdmf):
#     raise ValueError(f"Lungimea triangle_cells ({len(triangle_cells_xdmf)}) nu coincide cu lungimea triangle_domains ({len(triangle_domains_xdmf)})")

# # # Scrie fișierul .xdmf pentru plasă și domenii
# xdmf_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_nou.xdmf"
# meshio.write(
#     xdmf_file,
#     meshio.Mesh(
#         points=msh_to_xdmf.points,
#         cells=[("triangle", triangle_cells_xdmf)],
#         cell_data={"domains": [triangle_domains_xdmf]}
#     )
# )

# # # Citește plasa în FEniCSx
# with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
#     mesh = xdmf.read_mesh(name="Grid")
# # ########################################################################################
# # ##################################################################
# # #########################################################################################
# # # citește datele despre domenii din fișierul .h5
# h5_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.h5"
# with h5py.File(h5_file, "r") as h5:
#     domain_data = np.array(h5["/data2"], dtype=np.int32)

# # creează meshtags pentru domenii
# cell_indices = np.arange(len(triangle_domains_xdmf), dtype=np.int32)
# cell_tags = meshtags(mesh, mesh.topology.dim, cell_indices, domain_data)

# # verifică domeniile
# print("domenii unice (din .h5):", np.unique(cell_tags.values))
# # #################################################################################################
# # #####################################################################################################

# xdmf_path = f06_convert_msh_to_xdmf(msh_file) ### converteste msh in xdmf 
# xdmf_file = f07_extract_triangles_to_xdmf(msh_file) ### extrage doar tiunghiurile 
# # p04_plot_cell_tags(xdmf_file,title = "Fig. (4) Mesh in format .xdmf (pentru FEniCS)", cmap="Accent")
# domain, V = f08_inspect_xdmf_mesh(xdmf_file )
# domain = f09_read_xdmf_mesh(xdmf_file)
# domain = p06_visualize_xdmf_mesh(xdmf_file, title = "Fig. (6) Mesh in format .xdmf (pentru FEniCS)")

# # # # # #############################################################
# with XDMFFile(MPI.COMM_WORLD, xdmf_file , "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")
#     ct = xdmf.read_meshtags(domain, name="Grid")
# # plot_cell_tags(xdmf_file) # grafic verde, doar triunghiuri verzi 
# print("tri_domains shape:", tri_domains.shape)
# print("tris shape:", tris.shape)
# p10_plot_subdomains_tris(nodes, tris, domains) # domenii colorate , fara etichete 
# # # ##########################################################################################


# # # # valori materiale (corectate)
# mu0 = 4.0 * np.pi * 1e-7
# mu_r_FE = 4000
# mu_FE = mu_r_FE * mu0
# J0 = 3.4e6

# materials2 = {
#     0: (mu0,            0.0, "vacuum"),
#     1: (mu0,            0.0, "air"),
#     2: (mu0 * mu_r_FE,  0.0, "iron"),
# }


# mu, J = assign_materials(domain, materials2, domains)
# plot_materials_on_mesh(nodes, tris, domains, materials2)
# # # # # # ######################################################################################
# # # # # # #######################################################################################################################
# # # # # #######################################################################################################################
# # # # # # define the weak problem
# # # # # #######################################################################################################################
# # # # # # #######################################################################################################################
# with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_3dom_403_manual_z.xdmf", "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid")
#     ct = xdmf.read_meshtags(domain, name="Grid")   # <- foarte important#

# # ###########################################################################################
# VV = functionspace(domain, ("CG", 1))  # spațiu continuu
# uu = TrialFunction(VV)
# vv = TestFunction(VV)


# mu_val = 4000
# J_val = 1
# # mu_val = mu
# # J_val = J

# a = inner((1 / mu_val) * grad(uu), grad(vv)) * dx  # modificat pentru mu variabil
# L = inner(J_val, vv) * dx  # modificat pentru J variabil

# # a = inner((1.0 / mu_val) * grad(uu), grad(vv)) * dx
# # L = J_val * vv * dx
# # #########################################################################################

# # #########################################################################################
# # tdim = domain.topology.dim
# # fdim = tdim - 1
# # dirichlet_facets = locate_entities_boundary(domain, fdim, lambda x: ~np.isclose(x[1], 0.0))  # Dirichlet doar pe frontiera unde y != 0
# # dofs = locate_dofs_topological(VV, fdim, dirichlet_facets)
# # bc = dirichletbc(default_scalar_type(0), dofs, VV)

# # A_z = Function(VV)
# # problem = LinearProblem(a, L, u=A_z, bcs=[bc])
# # problem.solve()

# # Az_grid = pyvista.UnstructuredGrid(*vtk_mesh(VV))
# # Az_grid.point_data["A_z"] = A_z.x.array
# # Az_grid.set_active_scalars("A_z")

# # plotter = pyvista.Plotter()
# # plotter.add_mesh(Az_grid, cmap="Reds", show_edges=True, scalar_bar_args={"title":"A_z"})
# # plotter.view_xy()
# # plotter.show()

# #################################################################################################
# # definirea spațiului funcțional
# cell = "triangle"
# element = basix.ufl.element("Lagrange", cell, 1)
# vv = functionspace(mesh, element)
# uu = TrialFunction(vv)
# v = TestFunction(vv)

# # definirea măsurilor de integrare
# dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)

# # definirea parametrilor variabili
# mu0 = 4.0 * np.pi * 1e-7
# mu_r_fe = 4000
# mu_fe = mu_r_fe * mu0
# j0 = 3.4e6

# materials = {
#     2: (mu_fe, 0.0, "iron"),
#     1: (mu0, 0.0, "air"),
#     3: (mu0, j0, "copper"),
# }

# mu = Function(vv)
# j = Function(vv)

# # atribuie valori pentru mu și j pe noduri
# mu_values = {1: mu_fe, 2: mu0, 3: mu0}
# j_values = {1: 0.0, 2: 0.0, 3: j0}
# mu.x.array[:] = map_cell_tags_to_nodes(mesh, cell_tags, mu_values)
# j.x.array[:] = map_cell_tags_to_nodes(mesh, cell_tags, j_values)

# # forma variațională
# a = inner((1 / mu) * grad(uu), grad(v)) * dx(1) + \
#     inner((1 / mu) * grad(uu), grad(v)) * dx(2) + \
#     inner((1 / mu) * grad(uu), grad(v)) * dx(3)
# l = inner(j, v) * dx(1) + inner(j, v) * dx(2) + inner(j, v) * dx(3)

# # compilează formele
# a_form = form(a)
# l_form = form(l)

# # condiții la frontieră (dirichlet) V1 
# # tdim = mesh.topology.dim
# # fdim = tdim - 1
# # dirichlet_facets = locate_entities_boundary(mesh, fdim, lambda x: ~np.isclose(x[1], 0.0))
# # dofs = locate_dofs_geometrical(vv, lambda x: ~np.isclose(x[1], 0.0))
# # bc = dirichletbc(PETSc.ScalarType(0), dofs, vv)

# # condiții la frontieră (dirichlet) V2
# tdim = domain.topology.dim
# fdim = tdim - 1
# dirichlet_facets = locate_entities_boundary(domain, fdim, lambda x: ~np.isclose(x[1], 0.0))  # Dirichlet doar pe frontiera unde y != 0
# dofs = locate_dofs_topological(VV, fdim, dirichlet_facets)
# bc = dirichletbc(default_scalar_type(0), dofs, VV)

# # asamblarea matricii și vectorului
# a_matrix = assemble_matrix(a_form, bcs=[bc])
# a_matrix.assemble()
# b_vector = assemble_vector(l_form)
# apply_lifting(b_vector, [a_form], [[bc]])
# b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# set_bc(b_vector, [bc])

# # rezolvă sistemul liniar
# a_z = Function(vv)
# solver = PETSc.KSP().create(MPI.COMM_WORLD)
# solver.setOperators(a_matrix)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)
# solver.solve(b_vector, a_z.x.petsc_vec)
# #############################################################################################
# # === Vizualizare PyVista cu domenii și valori a_z ===
# az_grid = pyvista.UnstructuredGrid(*vtk_mesh(vv))

# # atașează valorile câmpului calculat (a_z) pe noduri
# az_grid.point_data["a_z"] = a_z.x.array.real
# az_grid.set_active_scalars("a_z")

# # aliniază corect domeniile cu celulele
# num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
# cell_indices = np.arange(num_cells, dtype=np.int32)
# domains_aligned = cell_tags.values[cell_indices]

# # atașează domeniile (ca cell_data)
# az_grid.cell_data["domain"] = domains_aligned

# # calculează centrele fiecărui triunghi
# cell_centers = az_grid.cell_centers().points

# # creează plotter-ul
# plotter = pyvista.Plotter()

# # desenează triunghiurile colorate după domeniu
# plotter.add_mesh(
#     az_grid,
#     scalars="domain",
#     cmap="tab10",
#     show_edges=True,
#     scalar_bar_args={"title": "Domain"},
# )

# # opțional: adaugă etichete pe triunghiuri (limitat la max_labels pentru viteză)
# max_labels = 200  # pune mai mic ca să nu blocheze
# for i, center in enumerate(cell_centers[:max_labels]):
#     plotter.add_point_labels(center, [str(domains_aligned[i])],
#                              text_color="black", font_size=8,
#                              always_visible=False)

# # etichete mari: câte una per domeniu, cu număr triunghiuri
# for dom in np.unique(domains_aligned):
#     mask = domains_aligned == dom
#     center = cell_centers[mask].mean(axis=0)
#     plotter.add_point_labels(center, [f"{dom} ({np.sum(mask)})"],
#                              text_color="black", font_size=16,
#                              always_visible=True)

# plotter.view_xy()
# plotter.show()

#####################
# import numpy as np
# import pyvista as pv
# from dolfinx.fem import Function, functionspace, TrialFunction, TestFunction, assemble_matrix, assemble_vector
# from dolfinx.fem.petsc import LinearProblem
# from mpi4py import MPI
# import ufl
# from ufl import inner, grad, as_vector, dx
# from dolfinx.plot import vtk_mesh
# import basix

dim = 2

# --- 1. Spațiu vectorial DG0 pentru B ---
# import basix
# import ufl
# from dolfinx.fem import FunctionSpace, Function, TrialFunction, TestFunction
# from ufl import as_vector, inner, dx

# cell_type = "triangle"
# element = basix.ufl.element("Lagrange", cell_type, 1)
# # spațiu vectorial 2D:
# element_vec = ufl.VectorElement(element, dim=2)  # două componente
# W = functionspace(mesh, element_vec)

# ##########################################################################################
# ##########################################################################################
# ##########################################################################################

import basix.ufl
from dolfinx.fem import Function, functionspace, form, assemble_matrix
from dolfinx.fem.petsc import assemble_vector
from ufl import TrialFunction, TestFunction, inner, dx, as_vector
from petsc4py import PETSc
from mpi4py import MPI

# definirea măsurii de integrare
dx = ufl.Measure("dx", domain=mesh)  # fără subdomain_data

# spațiu funcțional scalar pentru derivate
scalar_element = basix.ufl.element("Lagrange", "triangle", 1)
V_scalar = functionspace(mesh, scalar_element)

# spațiu funcțional vectorial pentru B
vector_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
W = functionspace(mesh, vector_element)

# proiecția derivatelelor lui A_z
dAz_dy = Function(V_scalar)  # ∂A_z/∂y
dAz_dx = Function(V_scalar)  # ∂A_z/∂x

# forme pentru proiecția derivatelelor
u_scalar = TrialFunction(V_scalar)
v_scalar = TestFunction(V_scalar)

# proiecția ∂A_z/∂y folosind integrare prin părți
a_dy = inner(u_scalar, v_scalar) * dx
L_dy = -inner(A_z, v_scalar.dx(1)) * dx  # derivata pe funcția de test
a_dy_form = form(a_dy)
L_dy_form = form(L_dy)
A_dy = assemble_matrix(a_dy_form)
with dAz_dy.x.petsc_vec.localForm() as dy_loc:
    dy_loc.set(0)
assemble_vector(dAz_dy.x.petsc_vec, L_dy_form)
solver_dy = PETSc.KSP().create(MPI.COMM_WORLD)
solver_dy.setOperators(A_dy)
solver_dy.setType(PETSc.KSP.Type.PREONLY)
solver_dy.getPC().setType(PETSc.PC.Type.LU)
solver_dy.solve(dAz_dy.x.petsc_vec, dAz_dy.x.petsc_vec)

# proiecția ∂A_z/∂x folosind integrare prin părți
a_dx = inner(u_scalar, v_scalar) * dx
L_dx = -inner(A_z, v_scalar.dx(0)) * dx  # derivata pe funcția de test
a_dx_form = form(a_dx)
L_dx_form = form(L_dx)
A_dx = assemble_matrix(a_dx_form)
with dAz_dx.x.petsc_vec.localForm() as dx_loc:
    dx_loc.set(0)
assemble_vector(dAz_dx.x.petsc_vec, L_dx_form)
solver_dx = PETSc.KSP().create(MPI.COMM_WORLD)
solver_dx.setOperators(A_dx)
solver_dx.setType(PETSc.KSP.Type.PREONLY)
solver_dx.getPC().setType(PETSc.PC.Type.LU)
solver_dx.solve(dAz_dx.x.petsc_vec, dAz_dx.x.petsc_vec)

# definirea funcțiilor și formelor pentru B
B = Function(W)
u_proj = TrialFunction(W)
v_proj = TestFunction(W)

# expresia B folosind derivatele proiectate
B_expr = as_vector((dAz_dy, -dAz_dx))

# forme pentru proiecția L2
a_proj = inner(u_proj, v_proj) * dx
L_proj = inner(B_expr, v_proj) * dx

# compilează formele
a_form = form(a_proj)
L_form = form(L_proj)

# asamblează matricea și vectorul
A = assemble_matrix(a_form)
with B.x.petsc_vec.localForm() as b_loc:
    b_loc.set(0)
assemble_vector(B.x.petsc_vec, L_form)

# configurează și rezolvă sistemul liniar
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.solve(B.x.petsc_vec, B.x.petsc_vec)

# eliberează memoria
A.destroy()
A_dy.destroy()
A_dx.destroy()
solver.destroy()
solver_dy.destroy()
solver_dx.destroy()
#############################################################################################
# --- 4. Pregătire PyVista ---
B_grid = pv.UnstructuredGrid(*vtk_mesh(mesh))
# magnitudine B pe noduri
B_values = B.x.array.reshape(-1, dim)
B_mag = np.linalg.norm(B_values, axis=1)
B_grid.point_data["|B|"] = B_mag
B_grid.set_active_scalars("|B|")

# --- 5. Săgeți (quiver) ---
cell_centers = B_grid.cell_centers()
vectors_cell = np.column_stack([B_values[:, 0], B_values[:, 1], np.zeros(len(B_values))])

# normalizare
norms = np.linalg.norm(vectors_cell[:, :2], axis=1, keepdims=True)
norms[norms == 0] = 1.0
vectors_norm = vectors_cell / norms
cell_centers.cell_data["Bnorm"] = vectors_norm

# Glyph cu lungime fixă
domain_size = max(B_grid.bounds[1]-B_grid.bounds[0], B_grid.bounds[3]-B_grid.bounds[2])
quiver = cell_centers.glyph(orient="Bnorm", scale=False, factor=0.01*domain_size)

# --- 6. Plot ---
plotter = pv.Plotter(window_size=[1200, 900])
plotter.add_mesh(B_grid, cmap="magma", show_edges=True, opacity=0.5,
                 scalar_bar_args={"title": "|B| (T)"})
plotter.add_mesh(quiver, color="red", opacity=0.9, show_scalar_bar=False)
plotter.view_xy()
plotter.camera.parallel_projection = True
plotter.add_title("Magnetic Field |B| and normalized arrows", font_size=14)
plotter.show_grid()
plotter.show()
