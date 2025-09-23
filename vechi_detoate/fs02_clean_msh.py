# infile = "steerer_steerer_fenics.msh"
# outfile = "steerer_steerer_fenics_clean.msh"

# infile = "comsol2dfara_spire_1pe8_vechi1.msh"
# outfile = "comsol2dfara_spire_1pe8_vechi1_clean.msh"


infile = "comsol2dfara_spire_toata_vechi1.msh"
outfile = "comsol2dfara_spire_toata_vechi1_clean.msh"



with open(infile, "r") as f_in, open(outfile, "w") as f_out:
    for line in f_in:
        # elimină tot ce vine după #
        clean = line.split("#")[0].strip()
        if clean:  # păstrează doar liniile ne-goale
            f_out.write(clean + "\n")

