import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import json

class myGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.shortest_path = None
        self.bidirectional = False
        self.plot_sp = False  # Flag for plotting Shortest Path
        self.plot_cc = False  # Flag for plotting Cluster Coefficients
        self.plot_no = False  # Flag for plotting Neighborhood Overlaps
    def read_wwwgraph(self, file_name):
        """Read a graph saved in JSON format by the spider program."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            for link in data['links']:
                source = link['source']
                target = link['target']
                #avoid self loops
                if source != target:
                    self.graph.add_edge(source, target)

    def read_graph(self, file_name): 
        """Read a graph G in memory from an external adjacent list File_name."""
        self.graph.clear()  # Clear existing graph
        self.shortest_path = None  # Empty shortest path
        unc_nodes = set()  # To keep track of nodes encountered
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    nodes = line.split()
                    source = int(nodes[0])
                    unc_nodes.add(source)
                    for target in nodes[1:]:
                        if target != '{}':
                            self.graph.add_edge(int(source), int(target))
                            unc_nodes.add(int(target))
        # Add any unconnected nodes to the graph
        for node in unc_nodes:
            if node not in self.graph.nodes():
                self.graph.add_node(node)

    def save_uni_graph(self, file_name):
        """Write the unidirectional graph in memory G to File_name using the adjacent list format."""
        with open(file_name, 'w') as file:
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                if neighbors:
                    file.write(f"{node} {' '.join(map(str, neighbors))}\n")
                else:
                    file.write(f"{node}\n")

    def save_bi_graph(self, file_name):
        """Write the bidirectional graph in memory G to File_name using the weighted adjacent list format."""
        with open(file_name, 'w') as file:
            for u, v, data in self.graph.edges(data=True):
                file.write(f"{u} {v} {data['a']} {data['b']}\n")

    def create_random_graph(self, n, c, random):
        """Create an Erdos-Renyi random graph G with n nodes and probability p = c(ln(n)/n)."""
        self.graph.clear()  # Clear existing graph
        self.shortest_path = None  # Empty shortest path
        
        # Create random graph
        if random:
            p = c * (math.log(n) / n)
            self.graph = nx.erdos_renyi_graph(n, p)
        else:
            self.graph = nx.karate_club_graph()

    def compute_shortest_path(self, source, target):
        """Compute the shortest path P between source and target."""
        self.shortest_path = None  # Empty shortest path
        if self.graph.has_node(int(source)) and self.graph.has_node(int(target)):
            try:
                self.shortest_path = nx.shortest_path(self.graph, source=int(source), target=int(target))
               # print(f"Shortest Path: {self.shortest_path}")
            except nx.NetworkXNoPath:
                print("No path found between the given nodes.")
        else:
            print("Source or target node not found in the graph.")

    def plot_graph(self):
        """Plot the graph G."""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Please create or read a graph first.")
            return
    
        pos = nx.spring_layout(self.graph)
        node_colors = ['red' if 'color' not in self.graph.nodes[node] else self.graph.nodes[node]['color'] for node in self.graph.nodes()]
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', node_color=node_colors)

        if self.plot_sp:  # Making shortest path stand out
            edges = [(self.shortest_path[i], self.shortest_path[i + 1]) for i in range(len(self.shortest_path) - 1)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges, width=2, edge_color='r', style='dashed')

        if self.plot_cc:  # Plot cluster coefficients
            cluster_coeffs = nx.clustering(self.graph)
            min_cluster_coeff = min(cluster_coeffs.values())
            max_cluster_coeff = max(cluster_coeffs.values())

            for node, coeff in cluster_coeffs.items():
                size = 100 + (coeff - min_cluster_coeff) * 154
                nx.draw_networkx_nodes(self.graph, pos, nodelist=[node], node_size=size, node_color=[(coeff / max_cluster_coeff, 1, 0)])

        if self.plot_no:  # Plot neighborhood overlaps
            for edge in self.graph.edges():
                if self.graph.has_edge(*edge):
                    nx.draw_networkx_edges(self.graph, pos, edgelist=[edge], width=2, edge_color='b', style='solid')

        plt.show()
    def assign_attributes(self, p):
        """
        Assign node attributes for homophily and edge signs for balanced graph.
        Parameters:
        p (float): Probability of assigning 'red' or 'blue' color to nodes,
        and assigning '+' or '-' sign to edges.
        """
        # Assign colors to nodes based on probability p
        for node in self.graph.nodes():
            self.graph.nodes[node]['color'] = 'red' if random.random() < p else 'blue'
        # Print out node attributes
        for node, data in self.graph.nodes(data=True):
            print(f"Node {node}: Color - {data['color']}")
        # Calculate assortativity (homophily)
        homophily = nx.attribute_assortativity_coefficient(self.graph, 'color')
        print("Homophily (Assortativity):", homophily)
        # Assign signs to edges using dwave_networkx
        for edge in self.graph.edges():
            if random.random() < p:
                self.graph.edges[edge]['sign'] = '-'
            else:
                self.graph.edges[edge]['sign'] = '+'
        # Determine if the graph is balanced using the cut ratio
        cut = {(u, v) for u, v, data in self.graph.edges(data=True) if data['sign']
== '-'}
        balanced_ratio = len(cut) / self.graph.number_of_edges()
        print("Balanced Graph Ratio (Size of Cut / Total Edges):", balanced_ratio)
        self.plot_cc = False
        self.plot_no = False
        self.plot_sp = False
        self.plot_graph()

    def partition_graph(self, num_components):
        """Remove edges with highest betweenness until the number of connected
        components is num_components."""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Please create or read a graph first.")
            return
        if self.graph.number_of_edges() == 0:
            print("Graph has no edges to remove.")    
            return
        while nx.number_connected_components(self.graph) > num_components:
            edge_betweenness = nx.edge_betweenness_centrality(self.graph)
            if not edge_betweenness: # Check if edge_betweenness is empty
                print("Edge betweenness is empty. Exiting partition.")
                break
            max_edge = max(edge_betweenness, key=edge_betweenness.get)
            self.graph.remove_edge(*max_edge)
        print("Partitioned graph into", num_components, "components.")
        self.assign_attributes(1) # Assign attributes after partitioning
        self.plot_graph() # Plot the updated graph


    def read_digraph(self, file_name):
        """Read a weighted directed graph G in memory from an external file in the format:
        source target a, b
        where a, b is the polynomial factor (a x + b) representing the weight."""
        self.graph.clear()  # Clear existing graph
        self.shortest_path = None  # Empty shortest path
        unc_nodes = set()  # To keep track of nodes encountered
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    nodes = line.split()
                    source = int(nodes[0])
                    unc_nodes.add(source)
                    for target in nodes[1:]:
                        if target != '{}':
                            self.graph.add_edge(int(source), int(target))
                            unc_nodes.add(int(target))
        # Add any unconnected nodes to the graph
        for node in unc_nodes:
            if node not in self.graph.nodes():
                self.graph.add_node(node)

    def plot_digraph(self):
        """Plot the directed graph G."""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Please create or read a graph first.")
            return

        pos = nx.spring_layout(self.graph)
        edge_labels = {(u, v): f"{data['a']}x + {data['b']}" for u, v, data in self.graph.edges(data=True)}
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()
    def compute_social_cost(self, pattern):
        """
        Compute the social optimality cost of the traffic pattern.
        """
        n = len(pattern)
        min_cost = float('inf')
        best_x = 0
        best_y = 0

        for x in range(n + 1):
            for y in range(n + 1):
                cost = x + (n - x) + (x - y) + (n - x + y)
                if cost < min_cost:
                    min_cost = cost
                    best_x = x
                    best_y = y

        return min_cost, best_x, best_y

    def compute_potential_energy(self, pattern):
        """
        Compute the potential energy of the traffic pattern.
        """
        energy = 0
        for driver in pattern:
            for i in range(len(pattern[driver]) - 1):
                edge = (pattern[driver][i], pattern[driver][i + 1])
                if self.graph.has_edge(*edge):
                    a = self.graph[edge[0]][edge[1]].get('a', 0)  # Default value for 'a' is 0 if not present
                    b = self.graph[edge[0]][edge[1]]['b']
                    energy += a * (pattern[driver].count(pattern[driver][i]) - 1) + b
        return energy

    def digraph_algos(self, source, dest, num_drivers):
        """
        Find the Nash equilibrium and social optimal when n drivers move from source to destination and plot the graph with the values

        """

        # Initialize traffic pattern
        pattern = {}
        for i in range(num_drivers):
            self.compute_shortest_path(source, dest)  # Compute shortest path for each driver
            path = self.shortest_path
            pattern[i] = path

        # Calculate costs for each driver
        nash_costs = {node: 0 for node in self.graph.nodes()}
        social_costs = {node: 0 for node in self.graph.nodes()}

        for driver in pattern:
            for i in range(len(pattern[driver]) - 1):
                edge = (pattern[driver][i], pattern[driver][i + 1])
                if self.graph.has_edge(*edge):
                    a = self.graph[edge[0]][edge[1]].get('a', 0) 
                    b = self.graph[edge[0]][edge[1]].get('b', 0)
                    cost_for_driver = a * (pattern[driver].count(pattern[driver][i]) - 1) + b
                    nash_costs[pattern[driver][i]] += cost_for_driver
                    social_costs[pattern[driver][i]] += a

        # Compute Total Nash Equilibrium Cost and Total Social Optimality Cost
        total_nash_cost = sum(nash_costs.values())
        total_soc_cost, best_x, best_y = self.compute_social_cost(pattern)

        # Plotting the graph with annotations
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)

        # Draw the edges
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', node_color='blue', node_size=700)

        # Draw the source and destination nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[int(source), int(dest)], node_color='yellow', node_size=700)

        # Write Nash Equilibrium Costs, Social Optimality Costs, and Edge Weights
        for (u, v, d) in self.graph.edges(data=True):
            edge_weight = f"{d['a']}x + {d['b']}"
            plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, edge_weight, color='black')
        for node in self.graph.nodes():
            nash_label = f"Nash: {nash_costs[node]}"
            soc_label = f"SOC: {social_costs[node]}"
            plt.text(pos[node][0] + 0.05, pos[node][1] + 0.05, nash_label, color='red', fontsize=10, verticalalignment='center')
            plt.text(pos[node][0] + 0.05, pos[node][1] - 0.05, soc_label, color='red', fontsize=10, verticalalignment='center')

        # Write number of drivers, source, and destination
        plt.text(pos[int(source)][0] + 0.15, pos[int(source)][1], f"Source: {source}", color='black', fontweight='bold')
        plt.text(pos[int(dest)][0] + 0.15, pos[int(dest)][1], f"Dest: {dest}", color='black', fontweight='bold')
        plt.text(0.85, 0.05, f"Drivers: {num_drivers}", color='black', fontweight='bold')
        # Calculate and write Total Nash Equilibrium Cost
        total_nash_cost = sum(nash_costs.values())
        plt.text(0.85, 0.10, f"Total Nash: {total_nash_cost}", color='black', fontweight='bold')

        # Calculate and write Total Social Optimality Cost
        plt.text(0.85, 0.2, f"Total SOC: {total_soc_cost}", color='black', fontweight='bold')

        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.show()

        print(f"\nTotal Social Cost: {total_soc_cost}")
        print(f"Total Nash Equilibrium: {total_nash_cost}")
    def plot_bipartite(self): 
        """Plot the bipartite graph"""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Please create or read a graph first.")
            return

        pos = nx.bipartite_layout(self.graph, [node for node, attr in self.graph.nodes(data=True) if attr['bipartite'] == 0])

        plt.figure(figsize=(10, 6))
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=500)

        plt.title("Bipartite Graph")
        plt.axis('off')
        plt.show()
    def create_bipartite(self, n, m, p):
        """Param: n: integer
                  Number of nodes in set A
                  m: integer
                  Number of nodes in set B
                  p: float`
                  The probability that the edge (u, v) is created where u in A and v in B
        Creates a random bipartite graph with node sets A and B and edge (u, v) exists with probability p where u in A and v in B"""
        if n <= 0 or m <= 0 or p < 0 or p > 1:
            print("Invalid parameters for bipartite graph creation.")
            return

        self.graph.clear()  # Clear existing graph
        self.shortest_path = None  # Empty shortest path

        nodes_a = range(1, n + 1)
        nodes_b = range(n + 1, n + m + 1)

        for node_a in nodes_a:
            for node_b in nodes_b:
                if random.random() < p:
                    self.graph.add_edge(node_a, node_b)

        # Assign bipartite attribute to nodes
        for node in nodes_a:
            self.graph.nodes[node]['bipartite'] = 0  # Set to 0 for set A
        for node in nodes_b:
            self.graph.nodes[node]['bipartite'] = 1  # Set to 1 for set B

        # Plot the bipartite graph
        self.plot_bipartite()

    def create_mcgraph(self, filename):
        """create market clearing graph
        Param:
            filename: string
            The file that contains the selling price and the valuation of each buyer as follows:"""
        self.graph.clear()  # Clear existing graph
        self.shortest_path = None  # Empty shortest path

        with open(filename, 'r') as file:
            lines = file.readlines()
            num_houses = int(lines[0].split()[0])

            # Parse house prices from the first line
            house_prices = list(map(int, lines[0].split()[1].split(',')))
            if len(house_prices) != num_houses:
                print("Error: Incorrect number of prices for houses")
                return

            # Create house nodes with prices
            for house_id, price in enumerate(house_prices, start=1):
                self.graph.add_node(house_id, bipartite=1, price=price)

            # Create buyer nodes with valuations
            for buyer_id, line in enumerate(lines[1:], start=1):
                valuations = list(map(int, line.strip().split(',')))
                if len(valuations) != num_houses:
                    print("Error: Incorrect number of valuations for buyer", buyer_id)
                    continue

                self.graph.add_node(buyer_id + num_houses, bipartite=0, valuation=dict(enumerate(valuations, start=1)))

                # Create edges between buyers and houses
                for house_id, valuation in enumerate(valuations, start=1):
                    self.graph.add_edge(house_id, buyer_id + num_houses, valuation=valuation)
        print("Market Clearing Graph Created.")
    def perfect_matching(self):
        """Compute the perfect matching for the bipartite graph."""
        if not nx.algorithms.bipartite.is_bipartite(self.graph):
            print("Graph is not bipartite. Cannot compute the perfect matching.")
            return
        # Get the bipartite sets
        bipartite_sets = nx.bipartite.sets(self.graph)
        buyer_nodes, house_nodes = bipartite_sets

        # Compute the perfect matching
        try:
            matching = nx.bipartite.maximum_matching(self.graph, top_nodes=buyer_nodes)
        except nx.NetworkXUnfeasible:
            print("No perfect matching exists for this graph.")
            return

        # Print the matching edges
        print("Perfect Matching Edges:")
        for buyer, house in matching.items():
            if buyer in buyer_nodes:
                print(f"Buyer {buyer} is matched with House {house}")

        # Plot the perfect matching
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)

        # Draw the edges
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', node_color='blue', node_size=700)

        # Draw the perfect matching edges in red
        for buyer, house in matching.items():
            if buyer in buyer_nodes:
                nx.draw_networkx_edges(self.graph, pos, edgelist=[(buyer, house)], edge_color='r', width=2.0)

        plt.show()

    def compute_psgraph(self):
        """
        Create a preferred-seller graph based on buyers and their preferred sellers.
        For a set of prices, construct an edge between each buyer and their preferred seller or sellers.
        """
        if not self.graph or not self.graph.nodes():
            print("Error: Graph is empty.")
            return None

        if not nx.bipartite.is_bipartite(self.graph):
            print("Error: Graph is not bipartite.")
            return None

        bipartite_sets = nx.bipartite.sets(self.graph)
        sellers = bipartite_sets[0]
        buyers = bipartite_sets[1]

        preferred_seller_graph = myGraph()

        for buyer in buyers:
            buyer_sellers = [neighbor for neighbor in self.graph.neighbors(buyer)]
            if buyer_sellers:
                for seller in buyer_sellers:
                    preferred_seller_graph.graph.add_edge(buyer, seller)

        # Output price and payoff of each buyer
        print("Preferred Seller Graph - Price and Payoff of Buyers:")
        for buyer in buyers:
            buyer_payoff = self.graph[buyer][list(self.graph[buyer].keys())[0]].get('valuation', 0)
            print(f"Buyer {buyer}: Payoff = {buyer_payoff}")

        return preferred_seller_graph
    def plot_psgraph(self):
        """Plot the preferred-seller graph (price and payoff of each buyer)."""
        if not self.graph or not self.graph.nodes():
            print("Error: Graph is empty.")
            return

        if not nx.bipartite.is_bipartite(self.graph):
            print("Error: Graph is not bipartite.")
            return

        pos = nx.bipartite_layout(self.graph, nx.bipartite.sets(self.graph)[0])

        prices = [self.graph.nodes[seller].get('valuation', 0) for seller in self.graph.nodes()]
        payoffs = [self.graph[seller][list(self.graph[seller].keys())[0]].get('valuation', 0) for seller in self.graph.nodes()]

        plt.figure(figsize=(10, 6))

        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.graph.nodes(), node_color='lightgreen', node_size=500)
        nx.draw_networkx_labels(self.graph, pos)

        for seller, price, payoff in zip(self.graph.nodes(), prices, payoffs):
            # Adjust the x and y offsets for better label positioning
            plt.text(pos[seller][0], pos[seller][1] + 0.05, f"Price: {price}", fontsize=10, ha='center', va='bottom')
            plt.text(pos[seller][0], pos[seller][1] - 0.05, f"Payoff: {payoff}", fontsize=10, ha='center', va='top')

        for edge in self.graph.edges(data=True):
            seller, buyer, valuation = edge
            plt.text((pos[seller][0] + pos[buyer][0]) / 2, (pos[seller][1] + pos[buyer][1]) / 2,
                     f"{valuation.get('valuation', 0)}", fontsize=8)

        plt.title("Preferred Seller Graph")
        plt.axis('off')
        plt.show()
    def pg_rank(self):
        """
        Compute the PageRank of the current weighted directed graph.
        """
        self.page_rank = nx.pagerank(self.graph, weight='a')  # Consider weight 'a' for PageRank computation

        file_name = "PageRank.txt"
        #Write the PageRank information to the file
        with open(file_name, 'w') as file:
            for node, rank in self.page_rank.items():
                file.write(f"{node}: {rank}\n")
    
        print(f"PageRank information saved to {file_name}")

    def plot_pg_rank(self, lower, upper):
        if not hasattr(self, 'page_rank'):
            print("PageRank not computed. Please run pg_rank() first.")
            return
        
        subgraph_nodes = [node for node, rank in self.page_rank.items() if lower <= rank <= upper]
        subgraph = self.graph.subgraph(subgraph_nodes)
        pos = nx.spring_layout(subgraph)
        
        plt.figure(figsize=(20, 10))
        nx.draw(subgraph, pos, with_labels=True, font_weight='bold')

        # Adjust label positions 
        label_pos = {k: (v[0], v[1] + 0.1) for k, v in pos.items()}
        nx.draw_networkx_labels(subgraph, label_pos, font_size=4)
        
        plt.title(f"Subgraph with PageRank between {lower} and {upper}")

        plt.show()

    def plot_lglg(self):
        """
        Plot the Log-log plot of the indegree for the weighted directed graph (www graph).
        """
        if not self.graph or not self.graph.nodes():
            print("Error: Graph is empty.")
            return

        indegrees = dict(self.graph.degree(weight='a'))
        indegree_values = list(indegrees.values())
        # Sort the indegree values in descending order
        indegree_values.sort(reverse=True)
        unique_indegrees, counts = np.unique(indegree_values, return_counts=True)

        # Plot the Loglog plot
        plt.figure(figsize=(10, 6))
        plt.loglog(unique_indegrees, counts, marker='o', linestyle='None', color='b', label='Data')

        # Fit a line to the data points
        coefficients = np.polyfit(np.log(unique_indegrees), np.log(counts), 1)
        line = np.exp(coefficients[1]) * unique_indegrees**coefficients[0]
        plt.loglog(unique_indegrees, line, color='r', label=f'Fit: {coefficients[0]:.2f}')

        plt.title("Log-log Plot of Indegree")
        plt.xlabel("Indegree (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plt.grid(True)
        plt.show()
    def cascade_effect(self, m, q):
        """
        Choose m (different) initiators from the nodes and simulate the cascade process
        in the graph with threshold q. Begin the cascade from each initiator and continue until the cascade
        is completed or no further nodes can be influenced. Plot the original graph with the initiators and
        the final graph with all the influenced nodes.
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Please create or read a graph first.")
            return

        initiators = random.sample(list(self.graph.nodes()), m)
        state = {node: 'B' for node in self.graph.nodes()}  # All nodes start with behavior B
        for initiator in initiators:
            state[initiator] = 'A'  # Initial nodes start with behavior A

        influenced_nodes = set(initiators)

        #propogation
        cascade_complete = False
        while not cascade_complete:
            cascade_complete = True
            for node in self.graph.nodes():
                if state[node] == 'B':
                    neighbor_behaviors = [state[neighbor] for neighbor in self.graph.neighbors(node)]
                    if neighbor_behaviors.count('A') / len(neighbor_behaviors) >= q:
                        state[node] = 'A'
                        influenced_nodes.add(node)
                        cascade_complete = False

        pos = nx.spring_layout(self.graph)  
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original graph with initiators
        nx.draw(self.graph, pos=pos, ax=axs[0], with_labels=True, node_color='lightblue', node_size=500)
        axs[0].set_title('Original Graph with Initiators')
        nx.draw_networkx_nodes(self.graph, pos=pos, ax=axs[0], nodelist=initiators, node_color='red', node_size=500)

        # Plot final graph with influenced nodes
        influenced_graph = nx.Graph(self.graph.subgraph(influenced_nodes))
        nx.draw(influenced_graph, pos=pos, ax=axs[1], with_labels=True, node_color='lightgreen', node_size=500)
        axs[1].set_title('Final Graph with Influenced Nodes')

        plt.show()

    def covid_sim(self, p, lifespan, shelter, r):
        """
        Initialize a small fraction of nodes as infected and track the progression of the epidemic over
        Lifespan using the SIR Model, Shelter-in-Place (reduce the number of directed edges by the fraction
        (shelter) and vaccinations effect in directed graphs. Contrast, analyze,
        and visualize (plot) the impact of different durations and timings of SIR model, shelter-in-place
        measures, and vaccination on the epidemic.
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Please create or read a graph first.")
            return

        # Initialize 
        infected_nodes = set(random.sample(list(self.graph.nodes()), int(p * self.graph.number_of_nodes())))
        susceptible_nodes = set(self.graph.nodes()) - set(infected_nodes)
        removed_nodes = set()
        
        # Initialize lists to track the number of nodes 
        susceptible_counts = []
        infected_counts = []
        removed_counts = []
        
        for day in range(lifespan):
            # Apply vaccination effect
            vaccinated_nodes = random.sample(list(susceptible_nodes), int(r * len(susceptible_nodes)))        
            susceptible_nodes -= set(vaccinated_nodes)

            # Spread the infection to neighbors of infected nodes
            new_infected_nodes = set()
            for node in infected_nodes:
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if random.random() > shelter:
                        # Apply shelter-in-place effect
                        if neighbor not in infected_nodes and neighbor in susceptible_nodes:
                            new_infected_nodes.add(neighbor)

            # Update infected nodes for the next day
            infected_nodes.update(new_infected_nodes)
            
            # Move infected nodes to removed stage after a certain period
            newly_removed = set()
            for node in infected_nodes:
                if random.random() < 0.1:  # Probability of being removed after infection period
                    newly_removed.add(node)
            removed_nodes.update(newly_removed)
            infected_nodes -= newly_removed

            # Track the number of nodes in each stage
            susceptible_counts.append(len(susceptible_nodes))
            infected_counts.append(len(infected_nodes))
            removed_counts.append(len(removed_nodes))

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(susceptible_counts)), susceptible_counts, marker='o', linestyle='-', label='Susceptible')
        plt.plot(range(len(infected_counts)), infected_counts, marker='o', linestyle='-', label='Infected')
        plt.plot(range(len(removed_counts)), removed_counts, marker='o', linestyle='-', label='Removed')
        plt.xlabel('Day')
        plt.ylabel('Number of Nodes')
        plt.title('Progression of Epidemic Over Time (SIR Model)')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    g = myGraph()
    while True:
        print("\nMenu:\n1. Read a Graph\n2. Read a Digraph\n3. Save the Graph\n4. Create a Graph\n5. Use an Algorithm\n6. Plot the Graph\n7. Assign and Validate Attributes\nx. Exit")
        user_choice = input("Enter your choice: ")

        if user_choice == '1':
            file_name = input("Enter file name to read the graph from: ")
            www = input("Is your graph a WWW graph?(Y/N)")
            if www == "N":
                g.read_graph(file_name)
            else:
                g.read_wwwgraph(file_name)
        elif user_choice == '2':
            file_name = input("Enter a filename to read the weighted directed graph from: ")
            g.read_digraph(file_name)
            g.bidirectional = True
        elif user_choice == '3':
            file_name = input("Enter file name to save graph to: ")
            if g.bidirectional == True:
                g.save_bi_graph(file_name)
            else:
                g.save_uni_graph(file_name)
        elif user_choice == '4':
            print("1. Random Erdos-Renyi Graph\n2. Karate-Club Graph\n3. Bipartite Graph\n4. Market-clearing\n")
            graphtype = input("Enter your choice: ")
            if graphtype == '1':
                n = int(input("Enter the number of nodes: "))
                c = float(input("Enter the constant c: "))
                g.create_random_graph(n, c, True)
            elif graphtype == '2':
                g.create_random_graph(0, 0, False)
            elif graphtype == '3':
                n = int(input("Enter the number of nodes in set A: "))
                m = int(input("Enter the number of nodes in set B: "))
                p = float(input("Enter the probability p where the edge (u,v) is created where u in A and v in B: "))
                g.create_bipartite(n, m, p)
            elif graphtype == '4':
                filename = input("Enter filename:")
                g.create_mcgraph(filename)
        elif user_choice == '5':
            print("1. Find the Shortest Path\n2. Partition the Graph\n3. Travel Equilibrium and Social Optimality\n4. Perfect Matching\n5. Preferred Seller graph\n6. Page Rank\n7. Cascade Effect\n8. COVID-19 Simulation\n")
            algo = int(input("Enter your Choice: "))
            if algo == 1:
                source = int(input("Enter source node: "))
                target = int(input("Enter target node: "))
                g.compute_shortest_path(source, target)
            elif algo == 2:
                num_components = int(input("Choose a number of components: "))
                g.partition_graph(num_components)
            elif algo == 3:
                source = input("Enter Source node: ")
                dest = input("Enter destination node: ")
                num_drivers = int(input("Input number of drivers: "))
                g.digraph_algos(source, dest, num_drivers)
            elif algo == 4:
                g.perfect_matching()
            elif algo == 5:
                g.compute_psgraph()
            elif algo == 6:
                g.pg_rank()
            elif algo == 7:
                m = int(input("Enter the number of initiators: "))
                q = float(input("Enter the threshold of the cascade: "))
                g.cascade_effect(m, q)
            elif algo == 8:
                p = float(input("Enter the fraction of initially infected nodes: "))
                lifespan = int(input("Enter the period of days of the simulation: "))
                shelter = float(input("Enter the fraction of edges that are not considered by the shelter-in-place: "))
                r = float(input("Enter the vaccination rate: "))
                g.covid_sim(p, lifespan, shelter, r)
        elif user_choice == '6':
            normal_plotting = True
            Toggle = 'Y'
            while Toggle == 'Y':
                print("1. Toggle the Shortest Path Switch\n2. Toggle the Cluster Coefficients\n3. Toggle Neighborhood Overlaps\n4. Toggle None\n5. Plot the Digraph\n6. Plot the Bipartite Graph\n7. Plot the Preferred Seller Graph\n8. Plot Page Rank\n9. Plot Loglog Plot\n")
                plot_choice = input("Enter your choice: ")
                if plot_choice == '1':
                    g.plot_sp = not g.plot_sp
                    if g.plot_sp:
                        if g.shortest_path is None:
                            print("The shortest path must be computed first.")
                            source = int(input("Enter source node: "))
                            target = int(input("Enter target node: "))
                            g.compute_shortest_path(source, target)
                        print("Enabled Shortest Path Plotting")
                    else:
                        print("Disabled Shortest Path Plotting")

                elif plot_choice == '2':
                    g.plot_cc = not g.plot_cc
                    if g.plot_cc:
                        print("Enabled Cluster Coefficient Plotting")
                    else:
                        print("Disabled Cluster Coefficient Plotting")

                elif plot_choice == '3':
                    g.plot_no = not g.plot_no
                    if g.plot_no:
                        print("Enabled Neighborhood Overlap Plotting")
                    else:
                        print("Disabled Neighborhood Overlap Plotting")
                elif plot_choice == '4':
                    Toggle = 'N'
                elif plot_choice == '5':
                    normal_plotting = False 
                    g.plot_digraph()
                elif plot_choice == '6':
                    normal_plotting = False
                    g.plot_bipartite()
                elif plot_choice == '7':
                    normal_plotting = False
                    g.plot_psgraph()
                elif plot_choice == '8':
                    normal_plotting = False
                    lower = float(input("Enter the lower bound of the ranks: "))
                    upper = float(input("Enter the upper bound of the ranks: "))
                    g.plot_pg_rank(lower, upper)
                elif plot_choice == '9':
                    normal_plotting = False
                    g.plot_lglg()
                Toggle = input("Toggle More? (Y/N)")
            if normal_plotting == True:       
                g.plot_graph()

        elif user_choice == '7':
            p = input("Enter probability p:")
            g.assign_attributes(float(p))
        elif user_choice == 'x':
            break
        else:
            print("Invalid input, try again.")
