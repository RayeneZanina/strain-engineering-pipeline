from pyvis.network import Network
import cobra

model = cobra.io.load_model("textbook") # only do textbook, anything else takes forever

genes = list(model.genes)
reactions = list(model.reactions)
metabolites = list(model.metabolites)

genes_to_idx = {gene.id: idx for idx, gene in enumerate(genes)}
rxn_to_idx = {rxn.id: idx for idx, rxn in enumerate(reactions)}
met_to_idx = {met.id: idx for idx, met in enumerate(metabolites)}

net = Network(
    height="900px",
    bgcolor="#000000",
    font_color="white",
    directed=True
)

for i, gene in enumerate(genes):
    net.add_node(
        f'genes_{i}',
        label=gene.id,
        color="red",
        shape="dot",
        title=f"Gene: {gene.name}"
    )

for i, rxn in enumerate(reactions):
    net.add_node(
        f"rxn_{i}",
        label=rxn.id,
        color="skyblue",
        shape="box",
        title=f"Reaction: {rxn.name}"
    )

for i, met in enumerate(metabolites):
    net.add_node(
        f"met_{i}",
        label=met.id,
        color="orange",
        shape="dot",
        title=f"Metabolite: {met.name}"
    )

for rxn in reactions:
    rxn_id = f"rxn_{rxn_to_idx[rxn.id]}"
    for gene in rxn.genes:
        gene_id = f"genes_{genes_to_idx[gene.id]}"
        net.add_edge(
            gene_id,
            rxn_id
        )
    for met, coeff in rxn.metabolites.items():
        if rxn.reversibility:
            met_id = f"met_{met_to_idx[met.id]}"
            net.add_edge(
                rxn_id,
                met_id
            )
            net.add_edge(
                met_id,
                rxn_id
            )
        elif coeff > 0:
            met_id = f"met_{met_to_idx[met.id]}"
            net.add_edge(
                rxn_id,
                met_id
            )
        elif coeff < 0:
            met_id = f"met_{met_to_idx[met.id]}"
            net.add_edge(
                met_id,
                rxn_id
            )
net.write_html("metabolic_network.html")