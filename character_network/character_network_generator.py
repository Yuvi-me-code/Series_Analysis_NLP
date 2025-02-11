import networkx as nx
from pyvis.network import Network
from itertools import chain
import pandas as pd
class CharacterNetworkGenerator():
    def __init__(self):
        pass

    def generate_character_network(self, data):
        windows = 10
        entity_relationship = []
        
        for row in data['ners']:
            previous_entity_in_window = []

            for sentence in row:
                previous_entity_in_window.extend(list(sentence))
                previous_entity_in_window = previous_entity_in_window[-windows:]

                # Ensure proper list flattening
                previous_entity_flattened = list(chain.from_iterable(
                    item if isinstance(item, list) else [item] for item in previous_entity_in_window
                ))

                for entity in sentence:
                    for entity_in_window in previous_entity_flattened:
                        if entity != entity_in_window:
                            entity_relationship.append(sorted([entity, entity_in_window]))

        relationship_data = pd.DataFrame({'value': entity_relationship})
        relationship_data['source'] = relationship_data['value'].apply(lambda x: x[0])
        relationship_data['target'] = relationship_data['value'].apply(lambda x: x[1])

        relationship_data = relationship_data.groupby(['source', 'target']).count().reset_index()
        relationship_data = relationship_data.sort_values('value', ascending=False)

        return relationship_data
    
    def draw_network_graph(self, relationship_data):
        relationship_data = relationship_data.sort_values('value', ascending=False)
        relationship_data = relationship_data.head(200)
        G = nx.from_pandas_edgelist(
            relationship_data,
            source = 'source',
            target = 'target',
            edge_attr='value',
            create_using=nx.Graph()
        )

        net = Network(notebook=True, width="1000px", height="700px", bgcolor="#222222", font_color="white", cdn_resources="remote")
        node_degree = dict(G.degree)

        nx.set_node_attributes(G, node_degree, 'size')
        net.from_nx(G)
        html = net.generate_html()
        html = html.replace("'", "\"")
        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        return output_html