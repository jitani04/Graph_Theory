# MyGraph - Graph Algorithms and Simulation Framework

## Overview

MyGraph is a Python-based framework built using the NetworkX library to manage, analyze, and simulate complex networks. It provides several features for graph creation, shortest path calculation, epidemic simulation using the SIR model, and visualizations. The framework is flexible and can handle different types of graphs, including random, bipartite, and real-world graphs.

## Features

- **Graph I/O**: Read/write graphs in adjacency list format or JSON format. Supports unidirectional and bidirectional (weighted) graphs.
  
- **Graph Creation**: Generate random Erdos-Renyi graphs and create real-world graphs such as the Karate Club graph or bipartite graphs.
  
- **Algorithms**: Compute the shortest path between two nodes, partition graphs into components, simulate market clearing and cascade effects, and simulate epidemic progression using the SIR model with shelter-in-place and vaccination measures.
  
- **Visualization**: Plot graphs using NetworkX and highlight shortest paths and other graph attributes.
  
- **COVID-19 Simulation**: Simulate the spread of an epidemic over time using the SIR model and analyze the effect of shelter-in-place policies and vaccinations.

## Usage

1. **Install Dependencies**: Ensure you have the following Python libraries installed:
   ```bash
   pip install networkx matplotlib numpy
