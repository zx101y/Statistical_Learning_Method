import networkx as nx
import matplotlib.pyplot as plt


dg = nx.DiGraph()
dg.add_edge(1, 2, val='青年', val2='有')
dg.add_node(1, val='年龄')

pos = nx.spring_layout(dg)
nx.draw(dg, pos, with_labels=True, node_size=2000)
node_labels = nx.get_node_attributes(dg, 'val')
nx.draw_networkx_labels(dg, pos, labels=node_labels)
nx.draw_networkx_edge_labels(dg, pos)
plt.show()
