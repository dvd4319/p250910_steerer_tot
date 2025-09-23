#######################################################
#### 1. Slide: Imports and Initialization ################
#######################################################
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl
from dolfinx import mesh, fem, plot
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem
import xml.etree.ElementTree as ET
# import sys
import pyvista

comm = MPI.COMM_WORLD

# Check package versions and sys.path for debugging
# if comm.rank == 0:
#     import dolfinx
#     import ufl
#     import basix
#     import pyvista
#     print(f"dolfinx version: {dolfinx.__version__}")
#     print(f"ufl version: {ufl.__version__}")
#     print(f"basix version: {basix.__version__}")
#     print(f"pyvista version: {pyvista.__version__}")
#     print("sys.path:", sys.path)

#######################################################
#### 2. Slide: Inspecting the XDMF Mesh File #############
#######################################################
# Function to inspect XDMF file
def inspect_xdmf(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print("XDMF file structure:")
        for grid in root.findall(".//Grid"):
            grid_name = grid.get("Name")
            print(f"  Grid: {grid_name}")
            for data_item in grid.findall(".//DataItem"):
                name = data_item.get("Name", "Unnamed")
                ref = data_item.text.strip() if data_item.text else "No reference"
                print(f"    DataItem: {name}, Reference: {ref}")
    except Exception as e:
        print(f"Error inspecting XDMF: {str(e)}")


#######################################################
#### 3. Slide: Reading the Mesh #############
#######################################################
# Import mesh from file
xdmf_file = "es02_capacitor_triangles.xdmf"
try:
    # Inspect the XDMF file
    if comm.rank == 0:
        inspect_xdmf(xdmf_file)

    with XDMFFile(comm, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        if comm.rank == 0:
            print(f"Mesh successfully read under name: Grid!")
            print(f"Number of cells (triangles): {domain.topology.index_map(domain.topology.dim).size_local}")

except Exception as e:
    if comm.rank == 0:
        print(f"Error reading XDMF file: {str(e)}")
    raise

#######################################################
#### 4. Slide: Boundary Definitions ######################
#######################################################

# Facet dimension (2D: fdim = dim - 1 = 1)
fdim = domain.topology.dim - 1

# Define boundary conditions using geometric functions
R1 = 0.2  # radius of the inner circle
R2 = 1.0  # radius of the outer circle

def inner_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), R1, rtol=1e-2)

def outer_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), R2, rtol=1e-2)

# Locate facets for boundary conditions
facets_inner = mesh.locate_entities_boundary(domain, fdim, inner_boundary)
facets_outer = mesh.locate_entities_boundary(domain, fdim, outer_boundary)

# Check if facets were found
if comm.rank == 0:
    print(f"Number of facets inner circle: {len(facets_inner)}")
    print(f"Number of facets outer circle: {len(facets_outer)}")

#######################################################
#### 5. Slide: Function Space and DOFs ###################
#######################################################
# Create function space
try:
    V = fem.functionspace(domain, ("Lagrange", 1))  # Lagrange, order 1
except TypeError:
    if comm.rank == 0:
        print("Warning: functionspace(domain, ('Lagrange', 1)) failed. Using basix.ufl.element.")
    element = basix.ufl.element("Lagrange", domain.ufl_cell().cellname(), 1)
    V = fem.functionspace(domain, element)

# Locate degrees of freedom for boundary conditions
dofs_inner = fem.locate_dofs_topological(V, fdim, facets_inner)
dofs_outer = fem.locate_dofs_topological(V, fdim, facets_outer)

# Check if dofs were found
if comm.rank == 0:
    print(f"Number of dofs inner circle: {len(dofs_inner)}")
    print(f"Number of dofs outer circle: {len(dofs_outer)}")

# Define Dirichlet boundary conditions
bc_inner = fem.dirichletbc(PETSc.ScalarType(1.0), dofs_inner, V)  # u=1 on inner circle
bc_outer = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_outer, V)  # u=0 on outer circle
bcs = [bc_inner, bc_outer]

#######################################################
#### 6. Slide: Variational Problem Setup #################
#######################################################

# Define variational formulation
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
eps = fem.Constant(domain, PETSc.ScalarType(1.0))
a = eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

# Solve the problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "ilu"})
uh = problem.solve()

#######################################################
#### 7. Slide: Error Computation #########################
#######################################################

# Compute L2 error (analytical solution: u(r) = (ln(R2/r)/ln(R2/R1)))
V2 = fem.functionspace(domain, ("Lagrange", 1))
uex = fem.Function(V2)
uex.interpolate(lambda x: np.log(R2 / np.sqrt(x[0]**2 + x[1]**2)) / np.log(R2 / R1))
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if comm.rank == 0:
    print(f"L1 error: {error_L2:.2e}")



#######################################################
#### 8. Slide: Saving Results ############################
#######################################################

# Save the solution
with XDMFFile(comm, "coaxial_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

if comm.rank == 0:
    print("Solution has been computed and saved in 'coaxial_solution.xdmf'.")


#######################################################
#### 9. Slide: Visualization with PyVista ################
#######################################################

# Visualization with pyvista
try:
    pyvista.OFF_SCREEN = True  # Force off-screen rendering
    pyvista.start_xvfb()  # Needed on Linux for headless environments
    if comm.rank == 0:
        print("Starting visualization with pyvista...")

    # Create VTK mesh for domain
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Visualize mesh
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.screenshot("es02_coaxial_mesh.png")
    if comm.rank == 0:
        print("Mesh saved in 'es02_coaxial_mesh.png'")

#######################################################
#### 10. Slide: Plotting the Solution #####################
#######################################################

    # Create VTK mesh for solution
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")

    # Visualize solution
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True, show_scalar_bar=True)
    u_plotter.view_xy()
    u_plotter.screenshot("es02_coaxial_solution.png")
    if comm.rank == 0:
        print("Solution saved in 'es02_coaxial_solution.png'")

except Exception as e:
    if comm.rank == 0:
        print(f"Error during visualization with pyvista: {str(e)}")

############################################################
#### 11. Slide: Comparing Numerical and Analytical Results #
############################################################


# Coordonatele nodurilor funcției uh (soluția numerică)
coords = V.tabulate_dof_coordinates()

# Valorile potențialului în noduri
u_vals = uh.x.array.real

# Exemplu: afișează primele 10 puncte cu coordonate și valoare
# if comm.rank == 0:
#     for i in range(min(10, len(u_vals))):
#         print(f"x = {coords[i]}, u = {u_vals[i]}")

##############################################################################################
if comm.rank == 0:
    u_analytical = []
    for i in range(len(u_vals)):
        r = np.sqrt(coords[i][0]**2 + coords[i][1]**2)
        u_analytical.append(np.log(R2 / r) / np.log(R2 / R1))

    for i in range(min(10, len(u_vals))):
        print(f"r = {np.sqrt(coords[i][0]**2 + coords[i][1]**2):.4f}, u_numeric = {u_vals[i]:.6f}, u_analytical = {u_analytical[i]:.6f}")

##############################################################################################
#### 12. Slide: Electric Field Lines Visualization ############################################
##############################################################################################

##############################################################################################
#### 13. Slide: Parallel Computation Benchmark ################################################
##############################################################################################
from time import perf_counter

if comm.rank == 0:
    print("\n=== Parallel Computation Benchmark ===")

# Creează un mesh mare pentru a genera multă muncă
N = 2000  # subdiviziuni pentru mesh
big_domain = mesh.create_unit_square(comm, N, N)

Vb = fem.functionspace(big_domain, ("Lagrange", 1))
ub = fem.Function(Vb)
x = big_domain.geometry.x
ub.x.array[:] = np.sin(np.pi*x[:,0]) * np.cos(np.pi*x[:,1])

vb = ufl.TestFunction(Vb)
a_big = ufl.inner(ub, vb) * ufl.dx

# Măsurăm timpul de asamblare
start = perf_counter()
A_big = fem.petsc.assemble_matrix(fem.form(a_big))
A_big.assemble()
end = perf_counter()

# Suma globală pentru a forța MPI communication
local_sum = np.sum(ub.x.array**2)
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

if comm.rank == 0:
    print(f"Assemble time: {end - start:.2f} s")
    print(f"Global sum check: {global_sum:.6e}")
    print("Monitorizează cu `htop` sau `top` pentru a vedea folosirea tuturor CPU cores.")
