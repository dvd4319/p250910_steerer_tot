from mpi4py import MPI
import numpy as np
from dolfinx import fem
from dolfinx.fem import (Function, functionspace, dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl import TrialFunction, TestFunction, dx, grad, dot
from dolfinx.mesh import locate_entities_boundary
from dolfinx import default_scalar_type

# --- 1. Load mesh from XDMF (exported from COMSOL) ---
with XDMFFile(MPI.COMM_WORLD, "comsol2dfara_spire_1pe8_vechi1_501_triangles.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")           # read mesh
    ct = xdmf.read_meshtags(domain, name="Grid")  # read domain tags for materials/boundaries

# --- 2. Define function space and trial/test functions ---
VV = functionspace(domain, ("CG", 2))  # quadratic continuous Galerkin space
uu = TrialFunction(VV)
vv = TestFunction(VV)

# --- 3. Material and source properties ---
mu_val = 4e-7 * 4000  # example: mu_FE = mu0 * mu_r
J_val = 3.4e6         # current density

# --- 4. Weak formulation of PDE ---
a = (1/mu_val) * dot(grad(uu), grad(vv)) * dx  # bilinear form
L = J_val * vv * dx                             # linear form

# --- 5. Boundary conditions ---
tdim = domain.topology.dim
facets = locate_entities_boundary(domain, tdim - 1, lambda x: np.full(x.shape[1], True))  # all boundary facets
dofs = locate_dofs_topological(VV, tdim - 1, facets)
bc = dirichletbc(default_scalar_type(0), dofs, VV)  # Dirichlet A_z = 0

# --- 6. Solve linear problem ---
A_z = Function(VV)
problem = LinearProblem(a, L, u=A_z, bcs=[bc])
problem.solve()
