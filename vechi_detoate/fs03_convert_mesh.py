def convert_custom_to_gmsh(infile, outfile):
    with open(infile, "r") as f:
        lines = f.readlines()

    clean_lines = []
    in_elements = False
    elements_buffer = []   # colectăm elementele reconstruite
    element_counter = 0

    for line in lines:
        # elimină comentarii
        line = line.split("#")[0].strip()
        if not line:
            continue

        if line.startswith("$Elements"):
            in_elements = True
            clean_lines.append(line)
            continue
        if line.startswith("$EndElements"):
            in_elements = False
            # înainte de a închide secțiunea scriem numărul de elemente și lista
            clean_lines.append(str(len(elements_buffer)))
            clean_lines.extend(elements_buffer)
            clean_lines.append(line)
            continue

        if not in_elements:
            # restul secțiunilor rămân la fel
            clean_lines.append(line)
        else:
            # suntem în $Elements
            parts = line.split()

            # sar peste linii cu text (edg, tri etc.)
            if any(p.isalpha() for p in parts):
                continue

            # sar peste linii cu un singur număr (număr de elemente inițial)
            if len(parts) == 1 and parts[0].isdigit():
                continue

            # altfel presupunem că e o linie de noduri
            element_counter += 1
            nodes = " ".join(parts)

            # determin tipul elementului după nr. de noduri
            if len(parts) == 2:      # muchie
                elm_type = 1
            elif len(parts) == 3:    # triunghi
                elm_type = 2
            else:
                raise ValueError(f"Format necunoscut: {line}")

            # elm-number elm-type number-of-tags tag1 tag2 node-list
            gmsh_line = f"{element_counter} {elm_type} 2 0 0 {nodes}"
            elements_buffer.append(gmsh_line)

    with open(outfile, "w") as f:
        f.write("\n".join(clean_lines))

    print(f"Fișier curat scris în: {outfile}")


# convert_custom_to_gmsh("steerer_steerer_fenics_clean.msh",
#                        "steerer_steerer_fenics_fixed.msh")


# convert_custom_to_gmsh("comsol2dfara_spire_1pe8_vechi1_clean.msh",
#                        "comsol2dfara_spire_1pe8_vechi1_fixed.msh")


convert_custom_to_gmsh("comsol2dfara_spire_toata_vechi1_clean.msh",
                       "comsol2dfara_spire_toata_vechi1_fixed.msh")
