#!/usr/bin/env python3
# solve_Az_from_xdmf.py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices
import sys

comm = MPI.COMM_WORLD

# -------------------------
# Config — editează după necesitate
# -------------------------
xdmf_file = "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf"
output_xdmf = "Az_solution_from_comsol.xdmf"

# Mapare tag -> material mu (HENRY/m)
# Ajustează valorile pentru materialele tale.
# Exemplu: tag 0 = air, 1 = iron, 2 = copper, ...
mu_map = {
    0: 4*np.pi*1e-7,     # air / vacuum
    1: 4e-5,             # iron (example smaller than real; ajustează)
    2: 1.26e-6,          # copper (approx mu0)
    3: 4*np.pi*1e-7,
    4: 4*np.pi*1e-7,
    5: 4*np.pi*1e-7
}

# Tag-urile celulelor care reprezintă bobina (curent). Ajustează la tag-urile tale.
coil_tags = [2]   # exemplu: tag 2 = coil; dacă ai mai multe, pune aici [2,3,...]

# Valoarea densității de curent în bobină (A/m^2). Ajustează după nevoie.
J_coil_value = 4.0e6

# Solver PETSc options (simple)
petsc_options = {
    "ksp_type": "cg",
    "pc_type": "ilu",
    "ksp_rtol": 1e-8
}

# -------------------------
# Read mesh + meshtags
# -------------------------
with io.XDMFFile(comm, xdmf_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    try:
        cell_tags = xdmf.read_meshtags(domain, name="Grid")
    except Exception:
        cell_tags = None
    # try read facet tags if present (some XDMF store facet meshtags separately)
    try:
        facet_tags = xdmf.read_meshtags(domain, name="Grid_facet")
    except Exception:
        facet_tags = None

if comm.rank == 0:
    print("Mesh loaded.")
    print("Topological dim:", domain.topology.dim, "Geometric dim:", domain.geometry.dim)
    if cell_tags is not None:
        unique = np.unique(cell_tags.values)
        print("Cell tags found:", unique)
    else:
        print("No cell_tags found in XDMF.")
    if facet_tags is not None:
        print("Facet tags found. Unique:", np.unique(facet_tags.values))
    else:
        print("No facet_tags found in XDMF.")

# Ensure connectivity for facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(tdim, fdim)

# -------------------------
# Function spaces
# -------------------------
V = fem.functionspace(domain, ("Lagrange", 1))   # continuous A_z
V0 = fem.functionspace(domain, ("DG", 0))        # piecewise constant per cell for mu and J

# -------------------------
# Build material (nu = 1/mu) and source J
# -------------------------
nu = fem.Function(V0)   # nu = 1/mu
J = fem.Function(V0)    # source J_z (per-cell)
# initialize
J.x.array[:] = 0.0

#################################

# coil_tags = lista de tag-uri pentru bobina ta
coil_tags = [2]  # ex: tag-ul 2 e bobina
J_value = 1e6    # densitatea de curent

# Setezi J pe celulele care au tag-ul bobinei
for tag in coil_tags:
    cells = cell_tags.find(tag)  # returnează indecșii local pentru proces
    J.x.array[cells] = J_value

#################################################
# If we have cell_tags, fill per tag; else use default mu_map[0]
if cell_tags is not None:
    # dolfinx MeshTags uses .find(tag) to get local cell indices
    tags_present = np.unique(cell_tags.values)
    for tag in tags_present:
        cells = cell_tags.find(tag)  # indices local to process
        mu_val = mu_map.get(int(tag), mu_map.get(0))
        if mu_val is None:
            raise RuntimeError(f"No mu value for tag {tag} and no default provided.")
        nu.x.array[cells] = 1.0 / float(mu_val)
        # set J in coil cells
        if int(tag) in coil_tags:
            J.x.array[cells] = float(J_coil_value)
else:
    # no tags — everything is air
    nu.x.array[:] = 1.0 / float(mu_map[0])
    # optionally set J via geometry if you want (not implemented)

# Broadcast some info
if comm.rank == 0:
    print("Material (nu) and J prepared.")
    print("J nonzero cells count (local):", np.count_nonzero(J.x.array))

# -------------------------
# Boundary facets / Dirichlet BC
# -------------------------
# priority 1: if facet_tags provided and you know tag for Dirichlet -> use it
dirichlet_facets = None

if facet_tags is not None:
    # If you have a specific numeric facet tag to use for Dirichlet, change here:
    # e.g. dirichlet_facet_tag = 1
    dirichlet_facet_tag = None
    # Try to auto-pick: if one tag corresponds to outer boundary maybe use max tag
    # If you want to force a specific tag, set above.
    if dirichlet_facet_tag is not None:
        dirichlet_facets = facet_tags.find(dirichlet_facet_tag)
    else:
        # fallback: consider facets with highest tag value (heuristic)
        uniq = np.unique(facet_tags.values)
        chosen = int(uniq.max())
        dirichlet_facets = facet_tags.find(chosen)
        if comm.rank == 0:
            print(f"Using facet tag {chosen} for Dirichlet (heuristic).")
else:
    # fallback: detect geometric boundary: y == 0 and x < 0 (negative x semi-axis)
    def on_negative_x_axis(x):
        # x has shape (gdim, npoints)
        xs = x[0, :]
        ys = x[1, :]
        eps = 1e-12
        mask = np.isclose(ys, 0.0, atol=1e-8) & (xs < 0.0 + 1e-12)
        return mask

    dirichlet_facets = locate_entities_boundary(domain, fdim, on_negative_x_axis)
    if comm.rank == 0:
        print("Detected dirichlet facets (geom) count (local):", len(dirichlet_facets))

# Locate DOFs
dofs_dir = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc_value = PETSc.ScalarType(0.0)  # A_z = 0 on that boundary
bc = fem.dirichletbc(bc_value, dofs_dir, V)

# -------------------------
# Variational forms
# -------------------------
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define measure dx with subdomain_data if cell_tags present (optional)
# if cell_tags is not None:
#     dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
# else:
#     dx = ufl.Measure("dx", domain=domain)

if cell_tags is not None:
    # acceptă și MeshTags, și ndarray
    if hasattr(cell_tags, "find"):
        tags_present = np.unique(cell_tags.values)
        for tag in tags_present:
            cells = cell_tags.find(tag)
            mu_val = mu_map.get(int(tag), mu_map.get(0))
            nu.x.array[cells] = 1.0 / mu_val
            if int(tag) in coil_tags:
                J.x.array[cells] = J_coil_value
    else:
        # e deja un ndarray
        tags_present = np.unique(cell_tags)
        for tag in tags_present:
            cells = np.where(cell_tags == tag)[0]
            mu_val = mu_map.get(int(tag), mu_map.get(0))
            nu.x.array[cells] = 1.0 / mu_val
            if int(tag) in coil_tags:
                J.x.array[cells] = J_coil_value

import ufl
from ufl import dx

# u = fem.Function(V)
# v = ufl.TestFunction(V)

# a = ufl.inner(nu * ufl.grad(u), ufl.grad(v)) * dx
# L = J * v * dx

from ufl import TrialFunction, TestFunction, inner, grad, dx

# --- Trial și Test functions ---
u = TrialFunction(V)   # variabila necunoscută (trial)
v = TestFunction(V)    # funcția de test

# --- Forme variationale ---
a = inner(nu * grad(u), grad(v)) * dx   # biliniar
L = J * v * dx                          # liniar

print("J min/max:", np.min(J.x.array), np.max(J.x.array))
print("Num cells with nonzero J:", np.count_nonzero(J.x.array))


# --- Problema ---
problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
uh = problem.solve()


# -------------------------
# Solve
# -------------------------
if comm.rank == 0:
    print("Assembling and solving linear system...")

if hasattr(cell_tags, "values"):
    tags_present = np.unique(cell_tags.values)
else:
    tags_present = np.unique(cell_tags)

problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
# problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options, form_compiler_parameters={"quadrature_degree": 2})
A_z = problem.solve()

if comm.rank == 0:
    print("Solve finished.")


# -------------------------
# Save mesh + solution
# -------------------------
with io.XDMFFile(MPI.COMM_WORLD, output_xdmf, "w") as xdmf_out:
    xdmf_out.write_mesh(domain)
    xdmf_out.write_function(A_z)
    # write also material and J as cell-functions for inspection
    try:
        # convert DG0 functions to cell tags style by writing them as functions
        xdmf_out.write_function(nu, "nu")
        xdmf_out.write_function(J, "J")
    except Exception:
        # some versions require other approach; ignore if fails
        pass

if comm.rank == 0:
    print("Wrote solution to:", output_xdmf)
# Dacă vrei să vizualizezi soluția
from dolfinx.io import XDMFFile

with XDMFFile(comm, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

#################################
import pyvista
from dolfinx.plot import vtk_mesh
import numpy as np
import ufl

# --- Plot A_z ---
plotter = pyvista.Plotter()
cells, types, points = vtk_mesh(V)
Az_grid = pyvista.UnstructuredGrid(cells, types, points)
Az_grid.point_data["A_z"] = A_z.x.array
Az_grid.set_active_scalars("A_z")

plotter.add_mesh(Az_grid, cmap="Reds", show_edges=True, scalar_bar_args={"title":"A_z"})
plotter.view_xy()
plotter.add_title("Magnetic Potential A_z", font_size=12)
plotter.show()

# --- Compute B = curl(A_z) ---
W = fem.FunctionSpace(domain, ("DG", 0, (domain.geometry.dim,)))
B = fem.Function(W)
B_expr = ufl.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)

# --- Plot B field vectors ---
top_imap = domain.topology.index_map(domain.topology.dim)
num_cells = top_imap.size_local + top_imap.num_ghosts
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
midpoints = mesh.compute_midpoints(domain, domain.topology.dim, np.arange(num_cells, dtype=np.int32))

values = np.zeros((num_cells, 3), dtype=np.float64)
values[:, :domain.geometry.dim] = B.x.array.real.reshape(num_cells, domain.geometry.dim)

cloud = pyvista.PolyData(midpoints)
cloud["B"] = values
glyphs = cloud.glyph("B", factor=2e6)

plotter = pyvista.Plotter()
plotter.add_mesh(glyphs)
plotter.view_xy()
plotter.add_title("Magnetic Flux Density B", font_size=12)
plotter.show()
