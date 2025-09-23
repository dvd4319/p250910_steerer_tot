import meshio

# mesh = meshio.read("steerer_steerer_fenics_fixed.msh")
# meshio.write("steerer_steerer_fenics.xdmf", mesh)


# mesh = meshio.read("mesh_valid.msh")
# meshio.write("mesh_valid_fenics.xdmf", mesh)



# mesh = meshio.read("comsol2dfara_spire_1pe8_vechi1_fixed.msh")
# meshio.write("comsol2dfara_spire_1pe8_vechi1_fixed.xdmf", mesh)


mesh = meshio.read("comsol2dfara_spire_toata_vechi1_fixed.msh")
meshio.write("comsol2dfara_spire_toata_vechi1_fixed.xdmf", mesh)


