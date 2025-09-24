from dolfinx.fem import functionspace, Function
import ufl
from ufl import as_vector, TrialFunction, TestFunction, inner, dx
from dolfinx.fem.petsc import LinearProblem

# --- 1. Create a vector function space DG0 (discontinuous per cell) ---
dim = 2  # physical space dimension
W = functionspace(domain, ("Discontinuous Lagrange", 0, (dim,)))  # DG0 vector space

B = Function(W)  # magnetic flux density B

# --- 2. Compute B = curl(A_z) in 2D ---
# UFL expression: B = (∂A_z/∂y, -∂A_z/∂x)
B_expr = as_vector([A_z.dx(1), -A_z.dx(0)])

# --- 3. Define trial and test functions for DG0 projection ---
u_proj = TrialFunction(W)
v_proj = TestFunction(W)

# --- 4. L2 projection forms ---
a_proj = inner(u_proj, v_proj) * dx      # bilinear form (mass matrix)
L_proj = inner(B_expr, v_proj) * dx      # linear form (right-hand side with curl)

# --- 5. Solve projection problem (DG0 has no BCs) ---
problem_proj = LinearProblem(a_proj, L_proj, u=B)  # solution stored in B
problem_proj.solve()
