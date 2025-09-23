code activate fenicsx 
gmsh es02_capacitor.geo -2 -format msh2 -o es02_capacitor.msh
meshio convert es02_capacitor.msh es02_capacitor.xdmf
------------------------------------------------------
conda activate fenicsx
#################################################
# la 2025 08 12 
pip install h5py
#################################################
### comentarii la data de 2025 08 13 
### PAS 1: din formatul .geo generez mesh-ul in formatele .msh si ulterior .xdmf:

gmsh es02_capacitor.geo -2 -format msh2 -o es02_capacitor.msh
meshio convert es02_capacitor.msh es02_capacitor.xdmf

### PAS 2: formatul generat nu are doar triunghiuri, si din aceasta cauza o sa dea erare 
### PAS 3: cu programul es00_generate_triangle.py generez fisierul:  es02_capacitor_triangles.xdmf care cica nu o sa aiba erori 
### PAS 4: fisierul es02_capacitor_triangles.xdmf  il import in es02_capacitor_numeric_comment.py si rezolv problema
### PAS 5: daca vreau sa aflu informatii desre grid apelez scriptul: es01_mesh_info.py 

#############################################################################################################################################################

code activate fenicsx 
gmsh es02_capacitor.geo -2 -format msh2 -o es02_capacitor.msh
meshio convert es02_capacitor.msh es02_capacitor.xdmf

conda activate fenicsx

# on 2025-08-12
pip install h5py

# comments on 2025-08-13

# STEP 1: from the .geo format I generate the mesh in the .msh format and later in .xdmf:
gmsh es02_capacitor.geo -2 -format msh2 -o es02_capacitor.msh
meshio convert es02_capacitor.msh es02_capacitor.xdmf

# STEP 2: the generated format does not contain only triangles, and because of this it will throw an error

# STEP 3: with the program es00_generate_triangle.py I generate the file es02_capacitor_triangles.xdmf, which supposedly will not have errors

# STEP 4: I import the file es02_capacitor_triangles.xdmf in es02_capacitor_numeric_comment.py and solve the problem

# STEP 5: if I want to get information about the grid, I run the script es01_mesh_info.py
