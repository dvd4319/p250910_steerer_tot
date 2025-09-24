from dolfinx.io.gmshio import read_from_msh
from dolfinx.io import XDMFFile
from dolfinx import mesh, plot
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
# import gmsh
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import matplotlib as mpl

from dolfinx import default_scalar_type, PETSc
from dolfinx.fem import (dirichletbc, Expression, Function, functionspace, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import compute_midpoints, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, as_vector, dot, dx, grad, inner

rank = MPI.COMM_WORLD.rank
gdim = 2
model_rank = 0
mesh_comm = MPI.COMM_WORLD


###########################################################
r = 0.1   # Radius of copper wires
R = 5     # Radius of domain
a = 1     # Radius of inner iron cylinder
b = 1.2   # Radius of outer iron cylinder
N = 8     # Number of windings
c_1 = 0.8 # Radius of inner copper wires
c_2 = 1.4 # Radius of outer copper wires
gdim = 2  # Geometric dimension of the mesh
#############################################################
MPI.COMM_WORLD.barrier()

mesh, ct, _ = read_from_msh("mg8_magnetic_structure1.msh", mesh_comm, gdim=2)


# grafic 1 
with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)

# "viridis", "plasma", "tab20", "Pastel1", "Accent"
Cmap = "Accent"
tdim = mesh.topology.dim
cells, types, x = plot.vtk_mesh(mesh)
grid = pyvista.UnstructuredGrid(cells, types, x)
num_local_cells = mesh.topology.index_map(tdim).size_local
grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
grid.set_active_scalars("Marker")
plotter = pyvista.Plotter()
plotter.add_text("Cell tags", font_size=12, color="black")
plotter.add_mesh(
    grid.copy(),
    scalars="Marker",
    cmap=Cmap,             # colormap mai deschis și contrastat
    show_edges=True,
    edge_color="black",      # margini negre pentru evidențierea triunghiurilor
    line_width=1,            # grosimea marginilor
    show_scalar_bar=True,
    opacity=1.0
)
plotter.view_xy()
plotter.link_views()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    pyvista.start_xvfb()
    plotter.screenshot("cell_tags.png", window_size=[800, 800])


# filename = "cell_tags.png"

# if not pyvista.OFF_SCREEN:
#     plotter.show(screenshot=filename)  # arată graficul și salvează imaginea
# else:
#     pyvista.start_xvfb()
#     plotter.screenshot(filename, window_size=[800, 800])


#######################################################################################################################
# Next, we define the discontinous functions for the permeability  mu and current density J_z using the MeshTags 
#######################################################################################################################

Q = functionspace(mesh, ("DG", 0))
material_tags = np.unique(ct.values)
mu = Function(Q)
J = Function(Q)
# As we only set some values in J, initialize all as 0
J.x.array[:] = 0
for tag in material_tags:
    cells = ct.find(tag)
    # Set values for mu
    if tag == 0:
        mu_ = 4 * np.pi*1e-7 # Vacuum
    elif tag == 1:
        mu_ = 1e-5 # Iron (This should really be 6.3e-3)
    else:
        mu_ = 1.26e-6 # Copper
    # mu.x.array[cells] = np.full_like(cells, mu_, dtype=ScalarType)
    mu.x.array[cells] = np.full_like(cells, mu_, dtype=default_scalar_type)
    if tag in range(2, 2+N):
        # J.x.array[cells] = np.full_like(cells, 1, dtype=ScalarType)
        J.x.array[cells] = np.full_like(cells, 1, dtype=default_scalar_type)
    elif tag in range(2+N, 2*N + 2):
        # J.x.array[cells] = np.full_like(cells, -1, dtype=ScalarType)
        J.x.array[cells] = np.full_like(cells, 1, dtype=default_scalar_type)
#######################################################################################################################
# In the code above, we have used a somewhat less extreme value for the magnetic permability of iron. 
# This is to make the solution a little more interesting. 
# It would otherwise be completely dominated by the field in the iron cylinder.
#######################################################################################################################
# We can now define the weak problem
#######################################################################################################################
V = functionspace(mesh, ("Lagrange", 1))
facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
dofs = locate_dofs_topological(V, tdim - 1, facets)
bc = dirichletbc(default_scalar_type(0), dofs, V)


u = TrialFunction(V)
v = TestFunction(V)
a = (1 / mu) * dot(grad(u), grad(v)) * dx
L = J * v * dx

#######################################################################################################################
# Solve the linear problem
#######################################################################################################################
A_z = Function(V)
problem = LinearProblem(a, L, u=A_z, bcs=[bc])
problem.solve()

#######################################################################################################################
# As we have computed the magnetic potential, we can now compute the magnetic field, by setting B=curl(A_z). Note that as we have chosen a function space of first order piecewise linear function to describe our potential, the curl of a function in this space is a discontinous zeroth order function (a function of cell-wise constants). We use dolfinx.fem.Expression to interpolate the curl into W.
#######################################################################################################################
W = functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
B = Function(W)
B_expr = Expression(as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)
#######################################################################################################################
# Note that we used ufl.as_vector to interpret the Python-tuple (A_z.dx(1), -A_z.dx(0)) as a vector in the unified form language (UFL).
# We now plot the magnetic potential A_z and the magnetic field B. 
# We start by creating a new plotter
#######################################################################################################################
# grafic 2 (potential magnetic vector A)


# Gradient pentru afișarea colormap-urilor
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Dicționar pentru salvarea colormap-urilor afișate
cmaps = {}

def plot_color_gradients(category, cmap_list):
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    for ax in axs:
        ax.set_axis_off()

    cmaps[category] = cmap_list

# Afișare gradient pentru colormap-ul "Greys"
plot_color_gradients('Sequential',
                     ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
plt.show()

# --- Partea PyVista ---

# Definește colormap-ul pentru plotare
CmapA = 'Reds'  

plotter = pyvista.Plotter()
Az_grid = pyvista.UnstructuredGrid(*vtk_mesh(V))
Az_grid.point_data["A_z"] = A_z.x.array
Az_grid.set_active_scalars("A_z")

plotter.add_mesh(
    Az_grid,
    cmap=CmapA,
    show_edges=True,
    scalar_bar_args={"title": "A_z", "vertical": True}
)

num_points = 10
step = max(1, Az_grid.n_points // num_points)

plotter.view_xy()
plotter.add_title("Magnetic Potential A_z", font_size=12)

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("Az_with_labels.png", window_size=[800, 800])


#######################################################################################################################
# Visualizing the magnetic field
# As the magnetic field is a piecewise constant vector field, we need create a custom plotting function. We start by computing the midpoints of each cell, which is where we would like to visualize the cell-wise constant vector. Next, we take the data from the function B, and shape it to become a 3D vector. We connect the vector field with the midpoint by using pyvista.PolyData.
#######################################################################################################################

#########################################
# # grafic 3 (magnetic flux density B) varianta originala 
plotter = pyvista.Plotter()
plotter.set_position([0, 0, 5])

top_imap = mesh.topology.index_map(mesh.topology.dim)
num_cells = top_imap.size_local + top_imap.num_ghosts
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
midpoints = compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32))

B_points = midpoints.copy()

num_dofs = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
assert (num_cells == num_dofs)
values = np.zeros((num_dofs, 3), dtype=np.float64)
values[:, :mesh.geometry.dim] = B.x.array.real.reshape(num_dofs, W.dofmap.index_map_bs)

cloud = pyvista.PolyData(midpoints)
cloud["B"] = values

glyphs = cloud.glyph("B", factor=2e6)
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    B_fig = plotter.screenshot("B.png")

#########################################
# # # grafic 3 (magnetic flux density B) varianta cu mai putine puncte 
# plotter = pyvista.Plotter()
# plotter.set_position([0, 0, 5])

# # # Obține topologia mesh-ului
# top_imap = mesh.topology.index_map(mesh.topology.dim)
# num_cells = top_imap.size_local + top_imap.num_ghosts
# mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

# # # Calculează midpoint-urile celulelor
# midpoints = compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32))

# B_points = midpoints.copy()

# # # Calculează valorile vectoriale B
# num_dofs = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
# assert (num_cells == num_dofs)
# values = np.zeros((num_dofs, 3), dtype=np.float64)
# values[:, :mesh.geometry.dim] = B.x.array.real.reshape(num_dofs, W.dofmap.index_map_bs)

# # Selectează 1 din fiecare 10 puncte pentru a evita aglomerarea
# step = 10
# sampled_points = midpoints[::step]
# sampled_values = values[::step]

# # Creează obiect PyVista pentru punctele eșantionate și adaugă vectorii B
# cloud = pyvista.PolyData(sampled_points)
# cloud["B"] = sampled_values

# # Generează glyph-urile pentru vectori (săgeți)
# glyphs = cloud.glyph("B", factor=2e6)

# # Adaugă mesh-ul wireframe
# plotter.add_mesh(grid, style="wireframe", color="k")

# # Adaugă glyph-urile în plotter
# plotter.add_mesh(glyphs)

# # Vizualizare
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     B_fig = plotter.screenshot("B.png")


#######################################################################################################################
# Datele pentru a calcula B analitic 
#######################################################################################################################

mu0 = 4 * np.pi * 1e-7
J_const = 1.0
S = np.pi * r * r

points_eval = B_points[::step]
print(f"points eval = {points_eval}")
dist_points = np.linalg.norm(points_eval, axis=1)

# Calculează câmpul magnetic B analitic în punctele alese
B_analitic = []
r22 = []

for pt in points_eval:
    x, y = pt[:2]
    r2 = np.sqrt(x**2 + y**2)
    if r2 < 1e-12:
        B_analitic.append([0.0, 0.0])
    else:
        factor = (mu0 * J_const * S) / (2 * np.pi * r2)
        B_analitic.append([-factor * y / r2, factor * x / r2])  # normalizezi y și x prin r2


B_analitic = np.array(B_analitic)

B_values = B.x.array.reshape((-1, 2))[::step]

B_norms = np.linalg.norm(B_analitic, axis=1)

errs_B = []
for Bn, Ba in zip(B_values, B_analitic):
    Bnrm = np.linalg.norm(Bn)
    Banrm = np.linalg.norm(Ba)
    errs_B.append(abs(Bnrm - Banrm) / (abs(Banrm) + 1e-20)*100)


print("\nIdx |    (x, y)      | Bx_num | By_num | |B|_num  | Bx_ana | By_ana | |B|_ana")
print("----------------------------------------------------------------------------------")
for i, (pt, Bn, Ba) in enumerate(zip(points_eval, B_values, B_analitic)):
    Bnrm = np.linalg.norm(Bn)
    Banrm = np.linalg.norm(Ba)
    # print(f"{i:3d} | ({pt[0]:7.3f},{pt[1]:7.3f}) | |{Bn[0]:7.2e} | {Bn[1]:7.2e} | {Bnrm:7.2e} | {Ba[0]:7.2e} | {Ba[1]:7.2e} | {Banrm:7.2e}")

print(f"\nB   errors: min={np.min(errs_B):.2f}%, max={np.max(errs_B):.2f}%, mean={np.mean(errs_B):.2f}%")

# print("\nIdx |    (x, y)      |  r_0 |   Bx ana       |     By ana       |   |B| analitic")
# print("----------------------------------------------------------------------------")
# for i, (pt, r_cu, Bv, Bn) in enumerate(zip(points_eval, r22, B_analitic, B_norms)):
#     print(f"{i:3d} | ({pt[0]:8.4f},{pt[1]:8.4f}) | {r_cu} |{Bv[0]:14.5e} | {Bv[1]:14.5e} | {Bn:14.5e}")




r_values = np.linalg.norm(points_eval[:, :2], axis=1)  # distanțele r = sqrt(x^2 + y^2)
B_norms_numeric = np.linalg.norm(B_values, axis=1)    # normele numerice ale vectorilor B

# Deja ai calculat B_analitic ca array de vectori, deci calculezi norma analitică
B_norms_analytic = np.linalg.norm(B_analitic, axis=1)


errors_abs = np.abs(B_norms_numeric - B_norms_analytic)
errors_rel = ((errors_abs / B_norms_analytic))*100

# print("\nIdx |   r [m] |  B_num [T] |  B_ana [T] | err abs | err rel [%]")
# print("----------------------------------------------------------------------------")
# for i, (r, Bn, Ba, ea, er) in enumerate(zip(r_values, B_norms_numeric, B_norms_analytic, errors_abs, errors_rel)):
#     print(f"{i} | {r:.4f} | {Bn:.5e} | {Ba:.5e} | {ea:.2e} | {er:.2%}")

print("\nIdx |    r [m]  | B_num [T]   | B_ana [T]   | err abs   | err rel [%]")
print("-------------------------------------------------------------------------")
for i, (r, Bn, Ba, ea, er) in enumerate(zip(r_values, B_norms_numeric, B_norms_analytic, errors_abs, errors_rel)):
    print(f"{i:02d}  | {r:8.4f} | {Bn:11.5e} | {Ba:11.5e} | {ea:9.2e} | {er:9.2f}")

# grafic 4 
# Convertim eroarea relativa în procente
errors_rel_percent = errors_rel

plt.figure(figsize=(8,5))
plt.plot(r_values, errors_rel_percent, 'o', color='red', label='Relative error [%]')
plt.xlabel('Distance r [m]')
plt.ylabel('Relative error [%]')
plt.title('Relative error of the magnetic field as a function of distance')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# import pyvista
# import numpy as np
# from dolfinx.plot import vtk_mesh

# cells, types, points = vtk_mesh(mesh)
# grid = pyvista.UnstructuredGrid(cells, types, points)

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, color='lightgrey', opacity=0.3, show_edges=True)

# point_cloud = pyvista.PolyData(points_eval)
# plotter.add_points(point_cloud, color='red', point_size=10)

# # Calculează r_0 = distanța față de origine
# r_0 = np.linalg.norm(points_eval[:, :2], axis=1)

# # Adaugă labeluri cu valoarea r_0 pentru fiecare punct
# for i, (point, r) in enumerate(zip(points_eval, r_0)):
#     pos = point + np.array([0.05, 0.05, 0])  # ușoară decalare pentru text
#     plotter.add_point_labels(pos, [f"{r:.2f}"], font_size=10, text_color='blue')

# plotter.view_xy()
# plotter.show()

#### grafic 5 
import pyvista
import numpy as np
from dolfinx.plot import vtk_mesh

# Extract mesh data (cells connectivity, cell types, and node coordinates) from dolfinx mesh for visualization
cells, types, points = vtk_mesh(mesh)
# Create a PyVista unstructured grid object from the mesh data
grid = pyvista.UnstructuredGrid(cells, types, points)
# Initialize the PyVista plotter window
plotter = pyvista.Plotter()
# Add the mesh grid to the plotter as a semi-transparent grey surface with visible edges
plotter.add_mesh(grid, color='lightgrey', opacity=0.3, show_edges=True)
# Create a PyVista point cloud object from the evaluation points (where numerical values were computed)
point_cloud = pyvista.PolyData(points_eval)

# Add these points to the plot with red color and a size of 10 pixels
plotter.add_points(point_cloud, color='red', point_size=10)
# Compute the distance r_0 of each evaluation point from the origin in the XY plane
r_0 = np.linalg.norm(points_eval[:, :2], axis=1)
# For each point and its corresponding distance r_0, add a label slightly offset from the point
for i, (point, r) in enumerate(zip(points_eval, r_0)):
    pos = point + np.array([0.05, 0.05, 0])  # Offset label position slightly in X and Y to avoid overlap with point
    plotter.add_point_labels(pos, [f"{r:.2f}"], font_size=10, text_color='blue')
# Set the camera to look directly down the Z-axis (top view on XY plane)
plotter.view_xy()
# Render and display the plot window
plotter.show()