import numpy as np
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from dolfinx import fem, mesh
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io import XDMFFile
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



def assign_material_properties(mesh, cell_tags, materials):
    """
    Asociază materialele și sursele de curent J pe celule.

    Parametri:
        mesh (dolfinx.mesh.Mesh): mesh-ul
        cell_tags (dolfinx.mesh.MeshTags): etichete pentru celule
        materials (dict): dict cu valori {tag: (mu_val, J_val)}

    Returnează:
        mu (fem.Function), J (fem.Function)
    """
    Q = fem.functionspace(mesh, ("DG", 0))
    mu = fem.Function(Q)
    J = fem.Function(Q)

    tags = np.unique(cell_tags.values)
    for tag in tags:
        cells = cell_tags.find(tag)
        if tag in materials:
            mu_val, J_val = materials[tag]
        else:
            mu_val, J_val = 4*np.pi*1e-7, 0.0  # fallback

        mu.x.array[cells] = mu_val
        J.x.array[cells] = J_val

    return mu, J


def define_boundary_conditions(mesh, V, value=0.0):
    """
    Definește condițiile de frontieră (Dirichlet pe y=0).
    Restul frontierelor rămân cu condiții naturale (Neumann).

    Parametri:
        mesh (dolfinx.mesh.Mesh): mesh-ul
        V (fem.FunctionSpace): spațiul pentru necunoscută
        value (float): valoarea pe Dirichlet

    Returnează:
        list cu obiecte fem.dirichletbc
    """
    facets_dirichlet = locate_entities_boundary(
        mesh, dim=1, marker=lambda x: np.isclose(x[1], 0.0, atol=1e-12))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = value

    bc = fem.dirichletbc(
        u_bc, fem.locate_dofs_topological(V, 1, facets_dirichlet))

    return [bc]

#########

############################################################################
################ (11) functie de materiale cu taguri   #####################
############################################################################

def assign_materials(domain, materials, domains):

    # definești spațiul pe mesh-ul efectiv (domain)
    Q = functionspace(domain, ("DG", 1))
    mu = Function(Q)
    J  = Function(Q)

    # domains = vectorul de taguri de la find_domains
    for tag, (mu_val, J_val, mat_val) in materials.items():
        cells = np.where(domains == tag)[0]   # toate celulele cu tag-ul respectiv
        mu.x.array[cells] = np.full_like(cells, mu_val, dtype=default_scalar_type)
        J.x.array[cells]  = np.full_like(cells, J_val,  dtype=default_scalar_type)

    return mu, J

#################################################
def assign_materials_variant(domain, materials, domains):

    mu0 = 4.0 * np.pi * 1e-7
    # --- creează spațiul DG0 pe mesh (valori pe celulă) ---
    Q = functionspace(domain, ("DG", 0))
    mu = Function(Q)
    J  = Function(Q)

    # inițializăm (default vacuum)
    mu.x.array[:] = default_scalar_type(mu0)
    J.x.array[:]  = default_scalar_type(0.0)

    # alocăm pe domenii
    for tag, (mu_val, J_val, name) in materials.items():
        cell_idxs = np.where(domains == tag)[0]
        if cell_idxs.size > 0:
            mu.x.array[cell_idxs] = default_scalar_type(mu_val)
            J.x.array[cell_idxs]  = default_scalar_type(J_val)

    # sumar în terminal
    print("\n=== Summary material assignment ===")
    for tag, (mu_val, J_val, name) in materials.items():
        idxs = np.where(domains == tag)[0]
        print(f"Tag {tag}: {name:14s} -> {len(idxs):4d} cells; sample indices: {idxs[:10]}")
######################################################
def plot_materials_on_mesh(nodes, tris, domains, materials, figsize=(10,10)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")

    triang = mtri.Triangulation(nodes[:,0], nodes[:,1], tris)

    # colorăm după domenii
    tpc = ax.tripcolor(triang, facecolors=domains, cmap="tab10", edgecolors="k", linewidth=0.3, alpha=0.8)

    # pentru fiecare domeniu, calculăm centrul și scriem materialul
    for tag, (mu_val, J_val, mat_val) in materials.items():
        cells = np.where(domains == tag)[0]
        if len(cells) == 0:
            continue
        
        # extragem nodurile triunghiurilor domeniului
        tri_nodes = tris[cells].reshape(-1)
        coords = nodes[tri_nodes]

        # centrul de masă al domeniului
        cx, cy = coords[:,0].mean(), coords[:,1].mean()

        # text pe grafic
        label = f"Tag {tag}\nμ={mu_val:.1e}\nJ={J_val:.1e} \n m = {mat_val}"
        ax.text(cx, cy, label, fontsize=12, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.title("Colored subdomains with assigned materials", fontsize = 16)
    plt.xlabel("x [m]", fontsize = 16)
    plt.ylabel("y [m]", fontsize = 16)
    plt.show()

######################################################

def plot_materials_on_mesh_variant(nodes, tris, domains, materials, figsize=(10,10)):
    triang = mtri.Triangulation(nodes[:,0], nodes[:,1], tris)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # color per triangle
    tpc = ax.tripcolor(triang, facecolors=domains, cmap="tab10",
                       edgecolors="k", linewidth=0.2, alpha=0.9)
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Domain ID")

    # pentru fiecare domeniu calculăm centrul de masă al nodurilor triunghiurilor sale
    for tag, (_,_,name) in materials.items():
        cells = np.where(domains == tag)[0]
        if cells.size == 0:
            continue
        tri_nodes = tris[cells].reshape(-1)          # repetări de noduri
        unique_nodes = np.unique(tri_nodes)          # noduri unice din domeniu
        coords = nodes[unique_nodes]                 # coordonate noduri din domeniu
        cx, cy = coords[:,0].mean(), coords[:,1].mean()
        ax.text(cx, cy, name, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title("Mesh colorat pe domenii (materialele etichetate în centru)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

