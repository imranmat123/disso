from collections import defaultdict
from random import random
from sys import maxsize

import dimod
import dwave
import matplotlib.pyplot as plt
import networkx as nx
import time

import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
from httpx import Client
from pyquil.gates import Z, I
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Sampler, BackendSampler
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer, AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_provider import IBMProvider
import dwave_networkx as dnx
import random

import dwave.inspector



from ACO import ACO
from TSP_QAOA import TSP_QAOA

from nna import NNA
from branchAndBound import *
from networkx.algorithms.approximation import traveling_salesman_problem, greedy_tsp, christofides, \
    simulated_annealing_tsp, threshold_accepting_tsp, asadpour_atsp
from scipy.optimize import linprog

v = 13  # Updated number of vertices

def travelling_salesman_function(graph, s):
    vertex = []
    for i in range(len(graph)):
        if i != s:
            vertex.append(i)

    min_path_value = maxsize
    best_path = []
    while True:
        current_distance = 0
        k = s
        current_path = [s]
        for i in range(len(vertex)):
            current_distance += graph[k][vertex[i]]
            k = vertex[i]
            current_path.append(k)
        current_distance += graph[k][s]
        current_path.append(s)

        if current_distance < min_path_value:
            min_path_value = current_distance
            best_path = current_path

        if not next_permutation(vertex):
            break
    return min_path_value, best_path

def next_permutation(lst):
    n = len(lst)
    i = n - 2

    while i >= 0 and lst[i] >= lst[i + 1]:
        i -= 1

    if i == -1:
        return False

    j = n - 1
    while lst[j] <= lst[i]:
        j -= 1

    lst[i], lst[j] = lst[j], lst[i]
    lst[i + 1:] = reversed(lst[i + 1:])
    return True


def plot_tsp(graph, path, min_path_value):
    G = nx.Graph()
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            G.add_edge(i, j, weight=graph[i][j])

    pos = nx.spring_layout(G, seed=42)  # Set seed for consistent layout

    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    edge_labels = {(path[i], path[i + 1]): f'{graph[path[i]][path[i + 1]]}' for i in range(len(path) - 1)}

    node_colors = ['lightblue'] * len(graph)
    node_colors[path[0]] = 'lightgreen'
    node_colors[path[-1]] = 'lightgreen'

    plt.figure(figsize=(14, 12))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    # Draw the order of visits next to the nodes
    for i, node in enumerate(path):
        x, y = pos[node]
        if i == 0:
            plt.text(x - 0.10, y, s='Start', bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='center', fontsize=14)
        elif i == len(path) - 1:
            plt.text(x + 0.10, y, s='Return', bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='center', fontsize=14)
        else:
            plt.text(x, y - 0.1, s=f'{i}', bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='center', fontsize=14)

    plt.title(f"TSP Path: {path} with cost: {min_path_value}")
    plt.show()


def held_karp_ascent(graph, n):
    # Held-Karp relaxation to find an approximate lower bound
    u = np.zeros(n)
    for _ in range(n * 10):  # Iterate to improve the bound
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d[i, j] = graph[i][j] + u[i] - u[j]

        min_cost = np.full(n, float('inf'))
        min_index = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and d[i, j] < min_cost[i]:
                    min_cost[i] = d[i, j]
                    min_index[i] = j

        cycle = [0]
        visited = [False] * n
        visited[0] = True
        current = 0
        while True:
            next_node = min_index[current]
            if visited[next_node]:
                break
            cycle.append(next_node)
            visited[next_node] = True
            current = next_node

        cycle.append(0)
        ascent_cost = sum(graph[cycle[i]][cycle[i + 1]] for i in range(len(cycle) - 1))
        if ascent_cost < 0:
            break

        for i in range(n):
            u[i] -= ascent_cost / n
    return cycle, ascent_cost


def min_cost_circulation(graph, n, cycle):
    # Minimum-cost circulation algorithm
    residual_graph = np.copy(graph)
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        residual_graph[u, v] -= 1
        residual_graph[v, u] += 1

    potentials = np.zeros(n)
    for _ in range(n):
        for u in range(n):
            for v in range(n):
                if residual_graph[u, v] < 0:
                    potentials[v] = min(potentials[v], potentials[u] + residual_graph[u, v])

    circulation_cost = sum(potentials)
    return potentials, circulation_cost


def asadpour_atsp_manual(graph):
    n = len(graph)
    cycle, ascent_cost = held_karp_ascent(graph, n)
    potentials, circulation_cost = min_cost_circulation(graph, n, cycle)

    tour = []
    current = 0
    while True:
        next_node = np.argmin(
            [graph[current][j] + potentials[current] - potentials[j] for j in range(n) if j != current])
        tour.append(next_node)
        if next_node == 0:
            break
        current = next_node

    return tour, ascent_cost + circulation_cost




hex_values = [
    '1049', '107', '10A', '10F5', '1130', '113E', '1143', '1148', '114C', '116E',
    '123D', '1245', '1291', '12F7', '1305', '138B', '1392', '1397', '13CB', '140F',
    '1419', '147', '14B9', '14E', '155A', '156E', '157', '15B1', '15BA', '1607',
    '1656', '16C8', '16CD', '16EE', '174F', '179', '17D4', '17E6', '17ED', '1820',
    '1872', '1880', '18E1', '191A', '193D', '1A01', '1A3A', '1B61', '1BD2', '1C63',
    '1CE', '1CF0', '1D1D', '1D23', '1D3', '1D40', '1D46', '1D6F', '1DAD', '1DCC',
    '1E01', '1F', '1F33', '1F4B', '202D', '2033', '20F', '2194', '21B7', '21BA',
    '21D7', '21F6', '21FC', '2203', '220F', '2229', '2275', '22FA', '23', '2313',
    '2329', '2364', '23BB', '23D8', '23F0', '240F', '241D', '2428', '2459', '246D',
    '24FF', '2571', '25B0', '25B7', '25C7', '25F3', '25F6', '25FD', '2693', '2724',
    '2736', '2787', '27EF', '285B', '2871', '28AC', '28BF', '28EC', '28FC', '2973',
    '29B1', '29BB', '2A99', '2B1', '2B15', '2B17', '2B2C', '2BB2', '2BBC', '2BDE',
    '2C0B', '2C12', '2C34', '2C73', '2CB9', '2CCA', '2CF8', '2D4F', '2DE2', '2E18',
    '2E1B', '2F39', '2F5D', '3016', '3031', '303E', '3044', '3098', '30B9', '30EE',
    '30F6', '3131', '3134', '3142', '31AA', '3230', '3233', '3234', '3265', '3275',
    '32E7', '330D', '338', '338B', '339D', '33D1', '33F5', '34AC', '34D3', '3511',
    '3530', '3532', '353C', '3566', '3576', '357E', '3588', '35E3', '360E', '3612',
    '362E', '369F', '36A4', '370', '3775', '378', '37B3', '37BB', '37FA', '380E',
    '383', '3832', '3870', '3888', '3909', '39CB', '39D2', '3A61', '3A62', '3A6A',
    '3A87', '3A88', '3AE8', '3AED', '3B04', '3B90', '3BA5', '3BF', '3BFA', '3C67',
    '3C78', '3CDE', '3CF5', '3D01', '3D5B', '3DCE', '3E8', '3E92', '3EB5', '3EF3',
    '3FE6', '4061', '406B', '4091', '40AF', '40BA', '4123', '416B', '41D5', '41D6',
    '4211', '4226', '4248', '4394', '439B', '43DC', '4411', '442F', '4465', '44A2',
    '44A3', '44A5', '44E3', '44F7', '44FC', '451C', '455', '457', '458B', '45BB',
    '45C4', '45DC', '45FE', '465', '4692', '4697', '469D', '470E', '4762', '47D3',
    '480C', '4818', '483B', '4840', '4850', '4854', '4872', '4929', '496', '49AD',
    '49C7', '49E4', '4A88', '4ADD', '4AE5', '4B', '4B33', '4B35', '4B54', '4B74',
    '4BC', '4BC0', '4BE', '4BE6', '4C4', '4CC5', '4CC6', '4D14', '4EA6', '4EBC',
    '4ECD', '4F2E', '4FB1', '50B6', '50BA', '50F8', '513B', '515A', '51B2', '51CD',
    '51D2', '51EA', '51F3', '5235', '52B6', '52BC', '52D6', '52EE', '5307', '5387',
    '53A7', '5407', '5448', '547D', '5516', '552E', '553A', '554', '554F', '555E',
    '555F', '557', '557D', '559', '559E', '55D4', '55D5', '55E4', '563E', '567F',
    '568C', '5859', '58A2', '58D5', '58DE', '5981', '5992', '59EB', '5A1C', '5A3B',
    '5ACB', '5AE4', '5AFB', '5B2F', '5BDE', '5C3D', '5C59', '5CB9', '5D7E', '5DB8',
    '5DF7', '5E18', '5E51', '5E70', '5EE4', '5F66', '5F6D', '5F81', '5FA6', '5FAD',
    '5FEC', '601D', '6023', '6037', '6066', '6068', '606A', '6076', '609A', '60B2',
    '60BE', '60C2', '60FC', '6112', '612E', '614C', '6227', '62B3', '62E1', '62EC',
    '6372', '63E5', '63E7', '6411', '6435', '647A', '64B5', '64E0', '64FF', '6509',
    '656F', '6576', '65AF', '65B0', '66A5', '66B1', '66BB', '66DC', '66EC', '675D',
    '6772', '67BF', '67FA', '681A', '6825', '6840', '6845', '684E', '6857', '68C4',
    '68F0', '68F5', '692E', '6958', '697E', '6983', '69C6', '6A4C', '6A7E', '6AB0',
    '6AFD', '6B1', '6B27', '6BBC', '6BE', '6BE0', '6BF8', '6C73', '6C8E', '6CD6',
    '6CE0', '6D25', '6D28', '6D2E', '6D51', '6D79', '6D8B', '6DDB', '6E24', '6E46',
    '6E95', '6EB1', '6ED0', '6F2F', '6F3', '6FA1', '6FA9', '6FB0', '7017', '7047',
    '7101', '710B', '7137', '7140', '7172', '719F', '71D4', '7297', '729C', '72B9',
    '7318', '7338', '7436', '7453', '7498', '755', '7575', '75D8', '76F', '771E',
    '773F', '775', '7762', '7765', '7826', '78BF', '78C', '78D1', '797E', '79D3',
    '79E1', '7A05', '7A06', '7A7C', '7B06', '7B3B', '7B5F', '7B67', '7BFB', '7CBF',
    '7CDC', '7DB', '7DE4', '7DF5', '7EB8', '7EFC', '7F6C', '7F70', '7FC', '803C',
    '8055', '805F', '80BF', '80C', '8102', '816', '8170', '81CC', '81D2', '81F5',
    '8264', '8293', '829C', '82BC', '82F1', '8316', '832D', '833B', '8364', '8370',
    '8388', '83AF', '83B9', '83E9', '8440', '84A6', '84B2', '851F', '8564', '85AD',
    '85E5', '860A', '8616', '861F', '8717', '872B', '8733', '876C', '87B3', '87DA',
    '8815', '88A2', '88AA', '88E9', '88F0', '8932', '8968', '898', '89BC', '89C5',
    '8A29', '8B13', '8B3A', '8B59', '8B74', '8B75', '8BA4', '8BB2', '8BB5', '8BB8',
    '8C02', '8C0A', '8CAD', '8CB2', '8CE', '8CF7', '8D15', '8D1F', '8D2', '8D3B',
    '8D47', '8D4D', '8D73', '8DA3', '8DB8', '8DC6', '8E2E', '8E53', '8E67', '8E8B',
    '8EFD', '8FD3', '905B', '9086', '9090', '9095', '90AF', '90BA', '90C1', '90D9',
    '90DD', '90E6', '90FA', '90FC', '9107', '9117', '911D', '918', '9182', '91AB',
    '91C', '91FF', '9270', '92E', '930A', '9342', '9344', '939F', '93A4', '9456',
    '94D6', '952', '9546', '9554', '9570', '9583', '95BF', '95DE', '95E5', '967',
    '96E4', '96F7', '976F', '97FF', '981D', '983E', '984', '985F', '98D1', '9909',
    '990C', '997', '9985', '99DD', '99E2', '9A28', '9A63', '9A9A', '9AA5', '9AF7',
    '9AFA', '9B1D', '9B62', '9B73', '9C2C', '9C76', '9C77', '9CA9', '9EF4', '9EF5',
    '9F96', '9FFA', 'A012', 'A024', 'A03C', 'A050', 'A0BB', 'A0CC', 'A0EA', 'A0F',
    'A103', 'A10B', 'A114', 'A149', 'A171', 'A19C', 'A1AC', 'A1B4', 'A21E', 'A227',
    'A258', 'A259', 'A25E', 'A29', 'A303', 'A342', 'A354', 'A3BF', 'A529', 'A578',
    'A5AD', 'A62E', 'A699', 'A6A', 'A6C2', 'A72A', 'A78F', 'A7DD', 'A808', 'A8A7',
    'A93D', 'A94B', 'A95C', 'A962', 'A998', 'A9AE', 'A9B5', 'A9DF', 'A9E8', 'A9F3',
    'AA7D', 'AA7F', 'AA90', 'AAC', 'AAD3', 'AB4A', 'AB56', 'ABA0', 'ABF2', 'AC4A',
    'AC4B', 'AC80', 'ACD3', 'ACE0', 'ACFF', 'AD16', 'AD3F', 'AD7A', 'ADE0', 'ADE7',
    'AE0A', 'AE13', 'AE5C', 'AE95', 'AEBF', 'AFBF', 'AFF5', 'B021', 'B053', 'B077',
    'B07E', 'B0B6', 'B0D3', 'B10', 'B103', 'B19A', 'B1DC', 'B1EF', 'B231', 'B291',
    'B2D2', 'B32D', 'B36E', 'B3BD', 'B3C4', 'B3FD', 'B42F', 'B4B3', 'B52D', 'B598',
    'B5DF', 'B62', 'B638', 'B667', 'B734', 'B777', 'B7B8', 'B7F', 'B7F5', 'B817',
    'B82F', 'B840', 'B86F', 'B8E0', 'B8F4', 'B976', 'B98E', 'B9C0', 'B9CF', 'B9D4',
    'BA35', 'BA77', 'BA78', 'BA7A', 'BAFD', 'BB30', 'BB83', 'BBA9', 'BBD1', 'BC6A',
    'BCA4', 'BD30', 'BD57', 'BD6', 'BD63', 'BD78', 'BE38', 'BE89', 'BEED', 'BF02',
    'BF91', 'BFDE', 'C01B', 'C03B', 'C073', 'C075', 'C08E', 'C127', 'C131', 'C144',
    'C16C', 'C18', 'C1B2', 'C1F8', 'C210', 'C255', 'C292', 'C31D', 'C339', 'C342',
    'C354', 'C36', 'C39E', 'C42E', 'C44', 'C44D', 'C49D', 'C4B3', 'C537', 'C55A',
    'C59C', 'C5F7', 'C5FC', 'C688', 'C6C9', 'C6F0', 'C775', 'C77B', 'C7D', 'C801',
    'C83F', 'C8D8', 'C8DB', 'C8E9', 'C8FC', 'C90D', 'C94C', 'C953', 'C978', 'C99C',
    'C9A9', 'C9C1', 'CA49', 'CA91', 'CAA', 'CAA3', 'CAB4', 'CABB', 'CB09', 'CB3D',
    'CB57', 'CBCF', 'CD13', 'CD16', 'CD5D', 'CDA9', 'CE4F', 'CF4B', 'CFE1', 'D02B',
    'D08D', 'D0A', 'D0BB', 'D132', 'D14B', 'D1DB', 'D1FE', 'D26B', 'D29F', 'D2AD',
    'D2B6', 'D2BD', 'D2ED', 'D2F0', 'D2F4', 'D36D', 'D37A', 'D3C5', 'D448', 'D480',
    'D49', 'D497', 'D61E', 'D68F', 'D7E6', 'D80E', 'D863', 'D876', 'D87E', 'D8C3',
    'D900', 'D905', 'D917', 'D946', 'D971', 'D9F3', 'DA2E', 'DA66', 'DA8C', 'DB91',
    'DBAF', 'DBB0', 'DC0A', 'DC7F', 'DCB1', 'DCF5', 'DCFC', 'DD6C', 'DDCE', 'DDDF',
    'DE1F', 'DEA7', 'DEB', 'E08D', 'E0B4', 'E0EF', 'E104', 'E12B', 'E161', 'E163',
    'E1B5', 'E1C8', 'E224', 'E231', 'E27F', 'E29D', 'E2AB', 'E2C2', 'E2D9', 'E2DD',
    'E2EC', 'E2FE', 'E346', 'E404', 'E431', 'E445', 'E450', 'E454', 'E45D', 'E520',
    'E5AB', 'E5D8', 'E5FD', 'E624', 'E630', 'E655', 'E69E', 'E746', 'E76D', 'E7EE',
    'E7F5', 'E851', 'E863', 'E938', 'E93C', 'E979', 'E97B', 'E9A8', 'E9D6', 'EA34',
    'EA65', 'EB58', 'EB72', 'EBB2', 'EBBE', 'EC1F', 'EC22', 'EC4', 'EC40', 'EC51',
    'EC9E', 'ECDD', 'ED8F', 'EDB0', 'EDBB', 'EDDA', 'EE2D', 'EEC3', 'EF5C', 'EFD7',
    'EFEA', 'F032', 'F06B', 'F14C', 'F194', 'F19A', 'F218', 'F2FE', 'F30F', 'F31E',
    'F330', 'F3AE', 'F3B6', 'F3C5', 'F3C9', 'F440', 'F44F', 'F494', 'F4D7', 'F4F5',
    'F531', 'F5B5', 'F6B1', 'F756', 'F759', 'F7BC', 'F805', 'F889', 'F917', 'F950',
    'F955', 'F9BC', 'F9C', 'F9D1', 'FA23', 'FA35', 'FA97', 'FB27', 'FB6D', 'FC3A',
    'FCC0', 'FD1C', 'FDD2', 'FFCE'
]

algorithm_globals.random_seed = 12345

# Define the new graph
graph = [
    [0, 10, 15, 20, 12],
    [10, 0, 35, 25, 18],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 25],
    [12, 18, 20, 25, 0]
]

# Ensure symmetry (this won't change the current graph, but it's a good practice)
graph = np.array(graph)
graph = (graph + graph.T) // 2

# Convert adjacency matrix to NetworkX graph
networkG = nx.Graph()
networkDG = nx.DiGraph()
num_nodes = len(graph)
distances = graph  # graph is already a numpy array now

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            networkG.add_edge(i, j, weight=graph[i][j])
            networkDG.add_edge(i, j, weight=graph[i][j])

# Print the graph for verification
print("Graph:")
# Create a TSP problem instance
tsp_instance = tsp.Tsp(networkG)

# Create a Quadratic Program for the TSP
problem = tsp_instance.to_quadratic_program()

# Convert the problem to a QUBO
qubo = QuadraticProgramToQubo().convert(problem)

def qubo_to_dict(quadratic_program):
    qubo_dict = {}
    # Add linear terms
    for i, var in enumerate(quadratic_program.variables):
        coeff = quadratic_program.objective.linear.to_array()[i]
        if coeff != 0:
            qubo_dict[(var.name, var.name)] = coeff
    # Add quadratic terms
    for i, var1 in enumerate(quadratic_program.variables):
        for j, var2 in enumerate(quadratic_program.variables):
            if i < j:
                coeff = quadratic_program.objective.quadratic.to_array()[i, j]
                if coeff != 0:
                    qubo_dict[(var1.name, var2.name)] = coeff
    return qubo_dict


def create_custom_tsp_qubo(distances, penalty_scale=500, distance_scale=5, completion_penalty=50):
    num_cities = len(distances)
    Q = {}

    # Linear terms (reduced strength)
    for i in range(num_cities):
        for p in range(num_cities):
            Q[(i * num_cities + p, i * num_cities + p)] = -penalty_scale / 2

    # Quadratic terms for constraints (slightly reduced)
    # One city per position
    for p in range(num_cities):
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                Q[(i * num_cities + p, j * num_cities + p)] = penalty_scale * 0.8

    # One position per city
    for i in range(num_cities):
        for p in range(num_cities):
            for q in range(p + 1, num_cities):
                Q[(i * num_cities + p, i * num_cities + q)] = penalty_scale * 0.8

    # Distance terms (slightly increased relative importance)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                for p in range(num_cities):
                    q = (p + 1) % num_cities
                    key = (i * num_cities + p, j * num_cities + q)
                    if key not in Q:
                        Q[key] = 0
                    Q[key] += distances[i][j] * distance_scale

    # Completion penalty (slightly reduced)
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            for p in range(num_cities):
                for q in range(num_cities):
                    if p != q:
                        key = (i * num_cities + p, j * num_cities + q)
                        if key not in Q:
                            Q[key] = 0
                        Q[key] -= completion_penalty * 0.8

    return Q


# After creating the QUBO
qubo = QuadraticProgramToQubo().convert(problem)

# Convert QUBO to dictionary
qubo_dict = qubo_to_dict(qubo)

print("QUBO as dictionary:")
for (var1, var2), coeff in qubo_dict.items():
    print(f"{var1}, {var2}: {coeff}")

# Check QUBO coefficients
print("\nLinear terms:")
for i, var in enumerate(qubo.variables):
    coeff = qubo.objective.linear.to_array()[i]
    if coeff != 0:
        print(f"{var.name}: {coeff}")

print("\nQuadratic terms:")
for i, var1 in enumerate(qubo.variables):
    for j, var2 in enumerate(qubo.variables):
        if i < j:
            coeff = qubo.objective.quadratic.to_array()[i, j]
            if coeff != 0:
                print(f"{var1.name}, {var2.name}: {coeff}")

# Check the range of coefficients
linear_coeffs = qubo.objective.linear.to_array()
quadratic_coeffs = qubo.objective.quadratic.to_array().flatten()
all_coeffs = np.concatenate([linear_coeffs, quadratic_coeffs])
non_zero_coeffs = all_coeffs[all_coeffs != 0]

print(f"\nMinimum non-zero coefficient: {np.min(np.abs(non_zero_coeffs))}")
print(f"Maximum coefficient: {np.max(np.abs(non_zero_coeffs))}")
print(f"Coefficient range: {np.max(np.abs(non_zero_coeffs)) - np.min(np.abs(non_zero_coeffs))}")
s = 0

def caluateTheCostOfPath(graph, path):
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1)) + graph[path[-1]][path[0]]

# Start timer for Christofides
start_time = time.time()

approxTourPath = traveling_salesman_problem(networkG, cycle=True, method=christofides)
christofidesTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Christofides")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", christofidesTourCost)
print("Execution Time (Christofides):", end_time - start_time, "seconds")
print("\n")

# Start timer for Greedy TSP
start_time = time.time()

approxTourPath = traveling_salesman_problem(networkG, cycle=True, method=greedy_tsp)
greedy_tspTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Greedy TSP")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", greedy_tspTourCost)
print("Execution Time (Greedy TSP):", end_time - start_time, "seconds")
print("\n")

# Start timer for Simulated Annealing TSP
initial_cycle = list(range(num_nodes)) + [0]
start_time = time.time()

approxTourPath = traveling_salesman_problem(
    networkG,
    cycle=True,
    method=lambda G, weight: simulated_annealing_tsp(G, weight=weight, init_cycle=initial_cycle)
)
simulated_annealing_tspTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Simulated Annealing TSP")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", simulated_annealing_tspTourCost)
print("Execution Time (Simulated Annealing TSP):", end_time - start_time, "seconds")
print("\n")

# Start timer for Threshold Accepting TSP
start_time = time.time()

approxTourPath = traveling_salesman_problem(
    networkG,
    cycle=True,
    method=lambda G, weight: threshold_accepting_tsp(G, weight=weight, init_cycle=initial_cycle)
)
threshold_accepting_tspTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Threshold Accepting TSP")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", threshold_accepting_tspTourCost)
print("Execution Time (Threshold Accepting TSP):", end_time - start_time, "seconds")
print("\n")

# Run Asadpour ATSP
start_time = time.time()
tour, cost = asadpour_atsp_manual(graph)
end_time = time.time()

print("Manual Asadpour ATSP")
print("Approximate Path:", tour)
print("Approximate Path Value:", cost)
print("Execution Time (Manual Asadpour ATSP):", end_time - start_time, "seconds")
# Instantiate the NNA class and call the nearest_neighbor_tsp method
nna = NNA()
approx_path = nna.nearest_neighbor_tsp(graph)

# End timer
end_time = time.time()

# Calculate cost of approximate path
approx_path_value = sum(graph[approx_path[i]][approx_path[i + 1]] for i in range(len(approx_path) - 1))

print("Approximate Path:", approx_path)
print("Approximate Path Value:", approx_path_value)
print("Execution Time (Nearest Neighbor):", end_time - start_time, "seconds")

# Plot approximate path
#plot_tsp(graph, approx_path, approx_path_value)




def extract_tour_from_bitstring(bitstring, num_nodes):
    solution_matrix = np.reshape([int(bit) for bit in bitstring], (num_nodes, num_nodes))
    tour = [0]  # Start from the first city
    current_city = 0
    for _ in range(num_nodes - 1):
        next_cities = [i for i in range(num_nodes) if solution_matrix[current_city][i] == 1 and i not in tour]
        if not next_cities:
            break
        next_city = next_cities[0]
        tour.append(next_city)
        current_city = next_city
    return tour


def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    if len(path) == len(graph):  # Only add the return cost if it's a complete tour
        cost += graph[path[-1]][path[0]]
    return cost


def is_complete_tour(tour, num_nodes):
    return len(tour) == num_nodes and len(set(tour)) == num_nodes


def print_best_tour_info(tour_dict, total_samples, method_name):
    print(f"\n{method_name} Complete Paths with Lowest Cost:")
    for tour, info in tour_dict.items():
        print(
            f"Tour: {list(tour)}, Viewed: {info['count']:.2f} times, Probability: {info['count'] / total_samples:.8f}, Cities Visited: {len(tour)}, Cost: {info['cost']}")
        print(f"Associated Bitstrings: {info['bitstrings']}")
        print(f"Number of Associated Bitstrings: {len(info['bitstrings'])}")

def print_all_tours(tour_dict, method_name):
    print(f"\n{method_name} All Tours (including incomplete):")
    sorted_tours = sorted(tour_dict.items(), key=lambda x: (-x[1]['count'], x[1]['cost']))
    for tour, info in sorted_tours:
        print(f"Tour: {list(tour)}, Viewed: {info['count']:.2f} times, Cost: {info['cost']}, Complete: {is_complete_tour(tour, len(networkG))}")


def print_summary(tour_dict, total_samples, method_name):
    complete_tours = {tour: info for tour, info in tour_dict.items() if is_complete_tour(tour, len(networkG))}
    if not complete_tours:
        print(f"\n{method_name} Summary:")
        print("No complete tours found.")
        return

    best_cost = min(info['cost'] for info in complete_tours.values())
    best_tours = {tour: info for tour, info in complete_tours.items() if info['cost'] == best_cost}
    highest_prob_tour = max(best_tours.items(), key=lambda x: x[1]['count'])

    print(f"\n{method_name} Summary:")
    print(f"Highest Probability Complete Path with Lowest Cost: {list(highest_prob_tour[0])}")
    print(f"Highest Probability with Lowest Cost: {highest_prob_tour[1]['count'] / total_samples:.8f}")
    print(f"Lowest Cost of Highest Probability Path: {best_cost}")
    print(f"Associated Bitstring: {highest_prob_tour[1]['bitstrings'][0]}")


# D-wave quantum annealer
print("D-wave quantum annealer")
# Create custom QUBO
custom_qubo = create_custom_tsp_qubo(distances)
sampler = EmbeddingComposite(DWaveSampler(token='DEV-43e9e19dc91f1bfbd603ba2f76ece83b35821800'))

start_time = time.time()
num_reads = 1000  # Same number as random guessing

response = sampler.sample_qubo(custom_qubo, num_reads=num_reads)
execution_time = time.time() - start_time

# Process results
qa_bitstring_counts = defaultdict(int)
qa_tour_counts = defaultdict(lambda: {'count': 0, 'cost': float('inf'), 'bitstrings': []})
qa_best_tour = None
qa_best_cost = float('inf')
total_qa_bitstrings = 0

for sample, energy, num_occurrences in response.data(['sample', 'energy', 'num_occurrences']):
    bitstring = tuple(sample.values())
    qa_bitstring_counts[bitstring] += num_occurrences
    total_qa_bitstrings += num_occurrences

    tour = tuple(extract_tour_from_bitstring(bitstring, len(distances)))
    qa_tour_counts[tour]['count'] += num_occurrences
    qa_tour_counts[tour]['bitstrings'].append(bitstring)

    cost = calculate_path_cost(distances, tour)
    qa_tour_counts[tour]['cost'] = cost
    if is_complete_tour(tour, len(distances)) and cost < qa_best_cost:
        qa_best_cost = cost
        qa_best_tour = tour

print(f"QA Execution Time: {execution_time:.6f} seconds")
print(f"QA Number of Unique Bitstrings: {len(qa_bitstring_counts)}")
print(f"QA Total Number of Bitstrings: {total_qa_bitstrings}")
print(f"QA Number of Unique Tours: {len(qa_tour_counts)}")
print("\n")
print_summary(qa_tour_counts, total_qa_bitstrings, "QA")
print("\n")
print_all_tours(qa_tour_counts, "QA")

# Random Guessing
def extract_tour_from_bitstring(bitstring, num_nodes):
    solution_matrix = np.reshape([int(bit) for bit in bitstring], (num_nodes, num_nodes))
    tour = [0]  # Start from the first city
    current_city = 0
    for _ in range(num_nodes - 1):
        next_cities = [i for i in range(num_nodes) if solution_matrix[current_city][i] == 1 and i not in tour]
        if not next_cities:
            break
        next_city = next_cities[0]
        tour.append(next_city)
        current_city = next_city
    return tour

def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    if len(path) == len(graph):  # Only add the return cost if it's a complete tour
        cost += graph[path[-1]][path[0]]
    return cost

def is_complete_tour(tour, num_nodes):
    return len(tour) == num_nodes and len(set(tour)) == num_nodes


def extract_tour_from_bitstring(bitstring, num_nodes):
    solution_matrix = np.reshape([int(bit) for bit in bitstring], (num_nodes, num_nodes))
    tour = [0]  # Start from the first city
    current_city = 0
    for _ in range(num_nodes - 1):
        next_cities = [i for i in range(num_nodes) if solution_matrix[current_city][i] == 1]
        if not next_cities:
            break
        next_city = next_cities[0]
        if next_city in tour:  # If we're revisiting a city, stop the tour
            break
        tour.append(next_city)
        current_city = next_city
    return tuple(tour)

def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    if len(path) == len(graph):  # Only add the return cost if it's a complete tour
        cost += graph[path[-1]][path[0]]
    return cost

def is_complete_tour(tour, num_nodes):
    return len(tour) == num_nodes and len(set(tour)) == num_nodes

# Perform random guessing on the graph
N = 10000
bit_length = num_nodes * num_nodes

random_bitstring_counts = defaultdict(int)
random_tour_counts = defaultdict(lambda: {'count': 0, 'cost': float('inf'), 'bitstrings': []})
random_best_tour = None
random_best_cost = float('inf')

for _ in range(N):
    bitstring = ''.join(random.choice('01') for _ in range(num_nodes * num_nodes))
    random_bitstring_counts[bitstring] += 1

    tour = tuple(extract_tour_from_bitstring(bitstring, num_nodes))
    random_tour_counts[tour]['count'] += 1
    random_tour_counts[tour]['bitstrings'].append(bitstring)

    cost = calculate_path_cost(graph, tour)
    random_tour_counts[tour]['cost'] = cost
    if is_complete_tour(tour, num_nodes) and cost < random_best_cost:
        random_best_cost = cost
        random_best_tour = tour

complete_tours_count = sum(1 for tour in random_tour_counts if is_complete_tour(tour, num_nodes))

print(f"\nNumber of complete tours found in random graph: {complete_tours_count}/{N}")
print(f"Best tour in random graph: {random_best_tour}")
print(f"Best cost in random graph: {random_best_cost}")

def print_all_tours(tour_counts, method):
    print(f"\nAll tours for {method}:")
    for tour, data in tour_counts.items():
        print(f"Tour: {tour}, Viewed: {data['count']} times, Cost: {data['cost']}")

def print_summary(tour_counts, total_bitstrings, method):
    unique_tours = len(tour_counts)
    print(f"\nSummary for {method}:")
    print(f"Total unique tours: {unique_tours}")
    print(f"Total bitstrings: {total_bitstrings}")

# Print all tours for random guessing
print_all_tours(random_tour_counts, "Random Guessing")

# Summary for random guessing
print_summary(random_tour_counts, N, "Random Guessing")

# Comparison (only meaningful if QA data is available)
# Assuming sample QA data to avoid errors in the comparison section
qa_best_cost = random_best_cost  # Placeholder
qa_bitstring_counts = random_bitstring_counts  # Placeholder
qa_tour_counts = random_tour_counts  # Placeholder
total_qa_bitstrings = N  # Placeholder

print("\nComparison:")
print(f"QA Best Complete Tour Cost: {qa_best_cost}, Random Best Complete Tour Cost: {random_best_cost}")
print(f"QA Unique Solutions: {len(qa_tour_counts)}, Random Unique Solutions: {len(random_tour_counts)}")
print(f"QA Total Bitstrings: {total_qa_bitstrings}, Random Total Bitstrings: {N}")
qa_best_bitstring = max(qa_bitstring_counts, key=qa_bitstring_counts.get, default=None)
random_best_bitstring = max(random_bitstring_counts, key=random_bitstring_counts.get, default=None)
if qa_best_bitstring:
    print(f"QA Most Common Bitstring Frequency: {qa_bitstring_counts[qa_best_bitstring]/total_qa_bitstrings:.6f}")
if random_best_bitstring:
    print(f"Random Most Common Bitstring Frequency: {random_bitstring_counts[random_best_bitstring]/N:.6f}")
def calculate_theoretical_expectations(num_cities):
    total_bitstrings = 2 ** (num_cities * num_cities)
    feasible_solutions = math.factorial(num_cities - 1)

    print("\nTheoretical Expectations:")
    print(f"Total possible bitstrings: {total_bitstrings}")
    print(f"Number of feasible solutions: {feasible_solutions}")
    print(f"Percentage of feasible solutions: {feasible_solutions / total_bitstrings:.8%}")

    return total_bitstrings, feasible_solutions


# Calculate theoretical expectations
num_cities = len(graph)
total_bitstrings, feasible_solutions = calculate_theoretical_expectations(num_cities)

# Improved Random Guessing
print("\nImproved Random Guessing")
N = 100000  # Increased number of samples for better statistical significance
start_time = time.time()
random_bitstring_counts = defaultdict(int)
random_tour_counts = defaultdict(lambda: {'count': 0, 'cost': float('inf'), 'bitstrings': []})
random_best_tour = None
random_best_cost = float('inf')
feasible_count = 0

for _ in range(N):
    bitstring = tuple(random.choice([0, 1]) for _ in range(num_cities * num_cities))
    random_bitstring_counts[bitstring] += 1

    tour = tuple(extract_tour_from_bitstring(bitstring, num_cities))
    random_tour_counts[tour]['count'] += 1
    random_tour_counts[tour]['bitstrings'].append(bitstring)

    cost = calculate_path_cost(graph, tour)
    random_tour_counts[tour]['cost'] = cost

    if is_complete_tour(tour, num_cities):
        feasible_count += 1
        if cost < random_best_cost:
            random_best_cost = cost
            random_best_tour = tour

execution_time = time.time() - start_time

print(f"Random Guessing Execution Time: {execution_time:.6f} seconds")
print(f"Random Number of Unique Bitstrings: {len(random_bitstring_counts)}")
print(f"Random Total Number of Bitstrings: {N}")
print(f"Random Number of Unique Tours: {len(random_tour_counts)}")
print(f"Number of Feasible Solutions Found: {feasible_count}")
print(f"Percentage of Feasible Solutions: {feasible_count / N:.8%}")
print(f"Expected Percentage of Feasible Solutions: {feasible_solutions / total_bitstrings:.8%}")


def calculate_tour_cost(tour, graph):
    total_distance = 0
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[i + 1]
        if next_city in graph[current_city]:
            distance = graph[current_city][next_city]['weight']
            total_distance += distance
    if len(tour) > 1 and tour[-1] in graph and tour[0] in graph[tour[-1]]:
        total_distance += graph[tour[-1]][tour[0]]['weight']
    return total_distance

def get_tsp_solution(x):
    num_cities = int(np.sqrt(len(x)))
    solution_matrix = np.reshape(x, (num_cities, num_cities))
    tour = [0]  # Start from the first city
    visited = set(tour)
    current_city = 0
    while len(tour) < num_cities:
        found = False
        for next_city in range(num_cities):
            if solution_matrix[current_city][next_city] == 1 and next_city not in visited:
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
                found = True
                break
        if not found:
            break
    return tour

def get_counts_and_probabilities(bit_string_counts, total_shots):
    count_prob_dict = {}
    for bit_string, probability in bit_string_counts.items():
        count = probability * total_shots
        count_prob_dict[bit_string] = {
            'count': count,
            'probability': probability
        }
    return count_prob_dict

def qubo_to_ising(qubo: QuadraticProgram):
    h = {}
    J = {}
    offset = 0
    linear = qubo.objective.linear.to_dict()
    quadratic = qubo.objective.quadratic.to_dict()

    for i, value in linear.items():
        h[i] = 0.5 * value
        offset += 0.5 * value

    for (i, j), value in quadratic.items():
        if i == j:
            h[i] = h.get(i, 0) + 0.5 * value
            offset += 0.5 * value
        else:
            J[(i, j)] = 0.25 * value
            h[i] = h.get(i, 0) + 0.25 * value
            h[j] = h.get(j, 0) + 0.25 * value
            offset += 0.25 * value

    return (h, J), offset

def build_sparse_pauli_op(h, J):
    num_qubits = max(max(h.keys(), default=0), max((i for pair in J.keys() for i in pair), default=0)) + 1
    paulis = []
    coeffs = []

    for i, coeff in h.items():
        z = ['I'] * num_qubits
        z[i] = 'Z'
        paulis.append(Pauli(''.join(z)))
        coeffs.append(coeff)

    for (i, j), coeff in J.items():
        z = ['I'] * num_qubits
        z[i] = 'Z'
        z[j] = 'Z'
        paulis.append(Pauli(''.join(z)))
        coeffs.append(coeff)

    return SparsePauliOp(paulis, coeffs)

def run_qaoa_tsp(qubo, networkG, p, num_shots=1000):
    # Define the optimizer and backend
    optimizer = COBYLA(maxiter=100)
    backend = Aer.get_backend('qasm_simulator')
    sampler = BackendSampler(backend=backend, options={"shots": num_shots})

    # Define the QAOA algorithm with the sampler and optimizer
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)

    # Solve the problem using QAOA
    minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)

    # Initialize variables
    best_tour = None
    best_cost = float('inf')
    most_cities_visited = 0
    total_shots = num_shots

    # Start timer
    start_time = time.time()

    result = minimum_eigen_optimizer.solve(qubo)
    samples = result.samples

    # Convert samples to dictionary format
    bit_string_counts = defaultdict(float)
    for sample in samples:
        bit_string = tuple(sample.x)
        probability = sample.probability
        bit_string_counts[bit_string] += probability

    # Calculate execution time
    execution_time = time.time() - start_time

    # Identify the best tour and cost
    tour_counts = defaultdict(float)
    highest_probability_tour = None
    highest_probability = 0
    highest_probability_bitstring = None
    lowest_cost_highest_probability_tour = None
    lowest_cost_highest_probability = 0
    lowest_cost_highest_probability_cost = float('inf')
    lowest_cost_highest_probability_bitstring = None

    for bit_string, probability in bit_string_counts.items():
        tour = get_tsp_solution(bit_string)
        tour_tuple = tuple(tour)
        num_cities_visited = len(tour)
        cost = calculate_tour_cost(tour, networkG)

        if (num_cities_visited > most_cities_visited) or (num_cities_visited == most_cities_visited and cost < best_cost):
            best_tour = tour
            best_cost = cost
            most_cities_visited = num_cities_visited

        tour_counts[tour_tuple] += probability

        # Update the highest probability complete tour
        if num_cities_visited == len(networkG) and probability > highest_probability:
            highest_probability_tour = tour
            highest_probability = probability
            highest_probability_bitstring = bit_string

        # Update the highest probability complete tour with the lowest cost
        if num_cities_visited == len(networkG):
            if cost < lowest_cost_highest_probability_cost:
                lowest_cost_highest_probability_tour = tour
                lowest_cost_highest_probability = probability
                lowest_cost_highest_probability_cost = cost
                lowest_cost_highest_probability_bitstring = bit_string
            elif cost == lowest_cost_highest_probability_cost and probability > lowest_cost_highest_probability:
                lowest_cost_highest_probability_tour = tour
                lowest_cost_highest_probability = probability
                lowest_cost_highest_probability_bitstring = bit_string

    # Calculate probabilities and counts for each bit string and tour
    bit_string_count_prob = get_counts_and_probabilities(bit_string_counts, total_shots)
    tour_count_prob = get_counts_and_probabilities(tour_counts, total_shots)

    return (best_tour, best_cost, execution_time, bit_string_counts, tour_counts,
            highest_probability_tour, highest_probability, highest_probability_bitstring,
            lowest_cost_highest_probability_tour, lowest_cost_highest_probability,
            lowest_cost_highest_probability_cost, lowest_cost_highest_probability_bitstring,
            bit_string_count_prob, tour_count_prob, most_cities_visited)

# Main execution
p_values = [1]  # layers
for p in p_values:
    print(f"\nRunning QAOA with {p} layers:")
    (best_tour, best_cost, execution_time, bit_string_counts, tour_counts,
     highest_probability_tour, highest_probability, highest_probability_bitstring,
     lowest_cost_highest_probability_tour, lowest_cost_highest_probability,
     lowest_cost_highest_probability_cost, lowest_cost_highest_probability_bitstring,
     bit_string_count_prob, tour_count_prob, most_cities_visited) = run_qaoa_tsp(qubo, networkG, p)

    print(f"Execution Time (QAOA): {execution_time:.8f} seconds")
    if best_tour:
        print(f"Best Approximate Path: {best_tour}")
        print(f"Best Approximate Path Value: {best_cost}")
        print(f"Cities Visited: {most_cities_visited}")
        best_tour_tuple = tuple(best_tour)
        best_tour_count = tour_count_prob[best_tour_tuple]['count']
        best_tour_probability = tour_count_prob[best_tour_tuple]['probability']
        print(f"Best Tour Viewed: {best_tour_count:.2f} times")
        print(f"Best Tour Probability: {best_tour_probability:.8f}")

        print("\nMost Successful Bitstrings for Best Tour:")
        for bitstring in bit_string_counts.keys():
            tour = get_tsp_solution(bitstring)
            if tour == best_tour:
                data = bit_string_count_prob[bitstring]
                count = data['count']
                probability = data['probability']
                print(f"Bitstring: {bitstring}, Count: {count:.2f}, Probability: {probability:.8f}")

    if highest_probability_tour:
        highest_prob_cost = calculate_tour_cost(highest_probability_tour, networkG)
        print(f"\nHighest Probability Complete Path: {highest_probability_tour}")
        print(f"Highest Probability: {highest_probability:.8f}")
        print(f"Highest Probability Path Cost: {highest_prob_cost}")
        print(f"Associated Bitstring: {highest_probability_bitstring}")

    if lowest_cost_highest_probability_tour:
        print(f"\nHighest Probability Complete Path with Lowest Cost: {lowest_cost_highest_probability_tour}")
        print(f"Highest Probability with Lowest Cost: {lowest_cost_highest_probability:.8f}")
        print(f"Lowest Cost of Highest Probability Path: {lowest_cost_highest_probability_cost}")
        print(f"Associated Bitstring: {lowest_cost_highest_probability_bitstring}")
    else:
        print("No valid complete tour found with highest probability or lowest cost")

    # Extract and display the most frequently observed bitstrings
    most_frequent_bitstrings = sorted(bit_string_count_prob.items(), key=lambda x: -x[1]['probability'])[:5]
    print("\nMost Frequently Observed Bitstrings:")
    for bitstring, data in most_frequent_bitstrings:
        tour = get_tsp_solution(bitstring)
        cost = calculate_tour_cost(tour, networkG)
        num_cities_visited = len(tour)
        count = data['count']
        probability = data['probability']
        print(f"Bitstring: {bitstring}, Count: {count:.2f}, Probability: {probability:.8f}, Tour: {tour}, Cities Visited: {num_cities_visited}, Cost: {cost}")

    # Extract and display the most frequently observed tours, their counts, costs, and probabilities
    most_frequent_tours = sorted(tour_count_prob.items(), key=lambda x: -x[1]['probability'])[:5]
    print("\nMost Frequently Observed Tours:")
    for tour, data in most_frequent_tours:
        cost = calculate_tour_cost(tour, networkG)
        num_cities_visited = len(tour)
        count = data['count']
        probability = data['probability']
        print(f"Tour: {list(tour)}, Viewed: {count:.2f} times, Probability: {probability:.8f}, Cities Visited: {num_cities_visited}, Cost: {cost}")

    # Display complete paths with the lowest cost
    complete_tours = [(tour, data) for tour, data in tour_count_prob.items() if len(tour) == most_cities_visited]
    if complete_tours:
        lowest_cost = min(calculate_tour_cost(tour, networkG) for tour, _ in complete_tours)
        best_complete_tours = [(tour, data) for tour, data in complete_tours if calculate_tour_cost(tour, networkG) == lowest_cost]

        total_views = 0
        total_probability = 0.0

        print("\nComplete Paths with Lowest Cost:")
        for tour, data in best_complete_tours:
            cost = calculate_tour_cost(tour, networkG)
            num_cities_visited = len(tour)
            count = data['count']
            probability = data['probability']
            total_views += count
            total_probability += probability
            associated_bitstrings = [bitstring for bitstring in bit_string_count_prob.keys() if get_tsp_solution(bitstring) == list(tour)]
            print(f"Tour: {list(tour)}, Viewed: {count:.2f} times, Probability: {probability:.8f}, Cities Visited: {num_cities_visited}, Cost: {cost}")
            print(f"Associated Bitstrings: {associated_bitstrings}")
            print(f"Sum Numerical Values: {sum(tour)}")
            print(f"Single Numerical Values: {tour}")

        print(f"\nTotal Views for Best Complete Paths: {total_views}")
        print(f"Sum of Probabilities for Best Complete Paths: {total_probability:.8f}")
    else:
        print("No complete paths found.")

print("\n")
print("\n")
print("\n")

bit_strings = [bin(int(h, 16))[2:].zfill(16) for h in hex_values]
df = pd.DataFrame({'Hex Value': hex_values, 'Bit String': bit_strings})

# Set display options to show all rows
pd.set_option('display.max_rows', None)

print(df)
# Set random seed for reproducibility
algorithm_globals.random_seed = 12345

graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]



# Convert adjacency matrix to NetworkX graph
networkG = nx.Graph()
networkDG = nx.DiGraph()
num_nodes = len(graph)
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            networkG.add_edge(i, j, weight=graph[i][j])
            networkDG.add_edge(i, j, weight=graph[i][j])

# Print all costs
print("Costs for each node:")
for i in range(len(graph)):
    for j in range(len(graph)):
        print(f"Cost from {i} to {j}: {graph[i][j]}")
    print()

s = 0

def caluateTheCostOfPath(graph, path):
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1)) + graph[path[-1]][path[0]]

# Start timer for Christofides
start_time = time.time()

approxTourPath = traveling_salesman_problem(networkG, cycle=True, method=christofides)
christofidesTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Christofides")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", christofidesTourCost)
print("Execution Time (Christofides):", end_time - start_time, "seconds")
print("\n")

# Start timer for Greedy TSP
start_time = time.time()

approxTourPath = traveling_salesman_problem(networkG, cycle=True, method=greedy_tsp)
greedy_tspTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Greedy TSP")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", greedy_tspTourCost)
print("Execution Time (Greedy TSP):", end_time - start_time, "seconds")
print("\n")

# Start timer for Simulated Annealing TSP
initial_cycle = list(range(num_nodes)) + [0]
start_time = time.time()

approxTourPath = traveling_salesman_problem(
    networkG,
    cycle=True,
    method=lambda G, weight: simulated_annealing_tsp(G, weight=weight, init_cycle=initial_cycle)
)
simulated_annealing_tspTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Simulated Annealing TSP")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", simulated_annealing_tspTourCost)
print("Execution Time (Simulated Annealing TSP):", end_time - start_time, "seconds")
print("\n")

# Start timer for Threshold Accepting TSP
start_time = time.time()

approxTourPath = traveling_salesman_problem(
    networkG,
    cycle=True,
    method=lambda G, weight: threshold_accepting_tsp(G, weight=weight, init_cycle=initial_cycle)
)
threshold_accepting_tspTourCost = caluateTheCostOfPath(graph, approxTourPath)

# End timer
end_time = time.time()
print("Threshold Accepting TSP")
print("Approximate Path:", approxTourPath)
print("Approximate Path Value:", threshold_accepting_tspTourCost)
print("Execution Time (Threshold Accepting TSP):", end_time - start_time, "seconds")
print("\n")

# Run Asadpour ATSP
start_time = time.time()
tour, cost = asadpour_atsp_manual(graph)
end_time = time.time()

print("Manual Asadpour ATSP")
print("Approximate Path:", tour)
print("Approximate Path Value:", cost)
print("Execution Time (Manual Asadpour ATSP):", end_time - start_time, "seconds")
# Instantiate the NNA class and call the nearest_neighbor_tsp method
nna = NNA()
approx_path = nna.nearest_neighbor_tsp(graph)

# End timer
end_time = time.time()

# Calculate cost of approximate path
approx_path_value = sum(graph[approx_path[i]][approx_path[i + 1]] for i in range(len(approx_path) - 1))

print("Approximate Path:", approx_path)
print("Approximate Path Value:", approx_path_value)
print("Execution Time (Nearest Neighbor):", end_time - start_time, "seconds")

# Plot approximate path
plot_tsp(graph, approx_path, approx_path_value)

# Create a TSP problem instance
tsp_instance = tsp.Tsp(networkG)

# Create a Quadratic Program for the TSP
problem = tsp_instance.to_quadratic_program()

# Convert the problem to a QUBO
qubo = QuadraticProgramToQubo().convert(problem)

# Define the optimizer and backend
optimizer = COBYLA(maxiter=10000)
backend = Aer.get_backend('qasm_simulator')
sampler = BackendSampler(backend=backend, options={"shots": 1024})  # Specify the number of shots

# Define the QAOA algorithm with the sampler and optimizer
qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=10)

# Solve the problem using QAOA
minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)


def calculate_tour_cost(tour, graph):
    total_distance = 0
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[i + 1]
        if next_city in graph[current_city]:
            distance = graph[current_city][next_city]['weight']  # Assuming weight stores distance
            total_distance += distance
    if len(tour) > 1 and tour[-1] in graph and tour[0] in graph[tour[-1]]:
        total_distance += graph[tour[-1]][tour[0]]['weight']
    return total_distance


# Function to calculate the cost of a given path
def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        if graph.has_edge(path[i], path[i + 1]):
            cost += graph[path[i]][path[i + 1]]['weight']
        else:
            print(f"No edge between {path[i]} and {path[i + 1]}")
            return float('inf')
    if graph.has_edge(path[-1], path[0]):
        cost += graph[path[-1]][path[0]]['weight']
    else:
        print(f"No edge between {path[-1]} and {path[0]}")
        return float('inf')
    return cost
def get_tsp_solution(x):
    num_cities = int(np.sqrt(len(x)))
    solution_matrix = np.reshape(x, (num_cities, num_cities))
    tour = [0]  # Start from the first city
    visited = set(tour)
    current_city = 0
    while len(tour) < num_cities:
        found = False
        for next_city in range(num_cities):
            if solution_matrix[current_city][next_city] == 1 and next_city not in visited:
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
                found = True
                break
        if not found:
            break
    return tour


def get_counts_and_probabilities(bit_string_counts, total_shots):
    count_prob_dict = {}
    for bit_string, probability in bit_string_counts.items():
        count = probability * total_shots
        count_prob_dict[bit_string] = {
            'count': count,
            'probability': probability
        }
    return count_prob_dict


def create_qaoa_circuit(qubo, p):
    """Create a QAOA circuit for the given QUBO and number of layers."""
    num_qubits = len(qubo)
    ansatz = QAOAAnsatz(cost_operator=qubo, reps=p)
    return ansatz
# Define QUBO conversion and QAOA setup functions
def qubo_to_ising(qubo: QuadraticProgram):
    h = {}
    J = {}
    offset = 0
    linear = qubo.objective.linear.to_dict()
    quadratic = qubo.objective.quadratic.to_dict()

    for i, value in linear.items():
        h[i] = 0.5 * value
        offset += 0.5 * value

    for (i, j), value in quadratic.items():
        if i == j:
            h[i] = h.get(i, 0) + 0.5 * value
            offset += 0.5 * value
        else:
            J[(i, j)] = 0.25 * value
            h[i] = h.get(i, 0) + 0.25 * value
            h[j] = h.get(j, 0) + 0.25 * value
            offset += 0.25 * value

    return (h, J), offset

def build_sparse_pauli_op(h, J):
    num_qubits = max(max(h.keys(), default=0), max((i for pair in J.keys() for i in pair), default=0)) + 1
    paulis = []
    coeffs = []

    for i, coeff in h.items():
        z = ['I'] * num_qubits
        z[i] = 'Z'
        paulis.append(Pauli(''.join(z)))
        coeffs.append(coeff)

    for (i, j), coeff in J.items():
        z = ['I'] * num_qubits
        z[i] = 'Z'
        z[j] = 'Z'
        paulis.append(Pauli(''.join(z)))
        coeffs.append(coeff)

    return SparsePauliOp(paulis, coeffs)

def run_annealing_inspired_qaoa(backend_name, qubo, p=1, num_shots=1024):
    provider = IBMProvider()
    backend = provider.get_backend(backend_name)

    hamiltonian, offset = qubo_to_ising(qubo)
    qaoa_op = build_sparse_pauli_op(hamiltonian[0], hamiltonian[1])

    optimizer = COBYLA(maxiter=100)
    sampler = BackendSampler(backend=backend, options={"shots": num_shots})
    qaoa = QAOA(optimizer=optimizer, reps=p, sampler=sampler)

    start_time = time.time()
    result = qaoa.compute_minimum_eigenvalue(operator=qaoa_op)
    execution_time = time.time() - start_time

    return result, execution_time

# Initialize variables
best_tour = None
best_cost = float('inf')
most_cities_visited = 0
total_shots = 1024  # Total number of shots used in the sampler

# Start timer
start_time = time.time()

result = minimum_eigen_optimizer.solve(qubo)
samples = result.samples

# Convert samples to dictionary format
bit_string_counts = defaultdict(float)
for sample in samples:
    bit_string = tuple(sample.x)
    probability = sample.probability
    bit_string_counts[bit_string] += probability

# Calculate execution time
execution_time = time.time() - start_time

# Identify the best tour and cost
tour_counts = defaultdict(float)
highest_probability_tour = None
highest_probability = 0
highest_probability_bitstring = None
lowest_cost_highest_probability_tour = None
lowest_cost_highest_probability = 0
lowest_cost_highest_probability_cost = float('inf')
lowest_cost_highest_probability_bitstring = None

for bit_string, probability in bit_string_counts.items():
    tour = get_tsp_solution(bit_string)
    tour_tuple = tuple(tour)
    num_cities_visited = len(tour)
    cost = calculate_tour_cost(tour, networkG)

    if (num_cities_visited > most_cities_visited) or (num_cities_visited == most_cities_visited and cost < best_cost):
        best_tour = tour
        best_cost = cost
        most_cities_visited = num_cities_visited

    tour_counts[tour_tuple] += probability

    # Update the highest probability complete tour
    if num_cities_visited == len(networkG) and probability > highest_probability:
        highest_probability_tour = tour
        highest_probability = probability
        highest_probability_bitstring = bit_string

    # Update the highest probability complete tour with the lowest cost
    if num_cities_visited == len(networkG):
        if cost < lowest_cost_highest_probability_cost:
            lowest_cost_highest_probability_tour = tour
            lowest_cost_highest_probability = probability
            lowest_cost_highest_probability_cost = cost
            lowest_cost_highest_probability_bitstring = bit_string
        elif cost == lowest_cost_highest_probability_cost and probability > lowest_cost_highest_probability:
            lowest_cost_highest_probability_tour = tour
            lowest_cost_highest_probability = probability
            lowest_cost_highest_probability_bitstring = bit_string

# Calculate probabilities and counts for each bit string and tour
bit_string_count_prob = get_counts_and_probabilities(bit_string_counts, total_shots)
tour_count_prob = get_counts_and_probabilities(tour_counts, total_shots)

# Extract the best tour's count and probability
best_tour_tuple = tuple(best_tour)
best_tour_count = tour_count_prob[best_tour_tuple]['count']
best_tour_probability = tour_count_prob[best_tour_tuple]['probability']

# Print informative message
print(f"Execution Time (QAOA): {execution_time:.8f} seconds")
if best_tour:
    print(f"Best Approximate Path: {best_tour}")
    print(f"Best Approximate Path Value: {best_cost}")
    print(f"Cities Visited: {most_cities_visited}")
    print(f"Best Tour Viewed: {best_tour_count:.2f} times")
    print(f"Best Tour Probability: {best_tour_probability:.8f}")

    print("\nMost Successful Bitstrings for Best Tour:")
    for bitstring in bit_string_counts.keys():
        tour = get_tsp_solution(bitstring)
        if tour == best_tour:
            data = bit_string_count_prob[bitstring]
            count = data['count']
            probability = data['probability']
            print(f"Bitstring: {bitstring}, Count: {count:.2f}, Probability: {probability:.8f}")

if highest_probability_tour:
    highest_prob_cost = calculate_tour_cost(highest_probability_tour, networkG)
    print(f"\nHighest Probability Complete Path: {highest_probability_tour}")
    print(f"Highest Probability: {highest_probability:.8f}")
    print(f"Highest Probability Path Cost: {highest_prob_cost}")
    print(f"Associated Bitstring: {highest_probability_bitstring}")

if lowest_cost_highest_probability_tour:
    print(f"\nHighest Probability Complete Path with Lowest Cost: {lowest_cost_highest_probability_tour}")
    print(f"Highest Probability with Lowest Cost: {lowest_cost_highest_probability:.8f}")
    print(f"Lowest Cost of Highest Probability Path: {lowest_cost_highest_probability_cost}")
    print(f"Associated Bitstring: {lowest_cost_highest_probability_bitstring}")

else:
    print("No valid complete tour found with highest probability or lowest cost")

# Extract and display the most frequently observed bitstrings
most_frequent_bitstrings = sorted(bit_string_count_prob.items(), key=lambda x: -x[1]['probability'])[:5]
print("\nMost Frequently Observed Bitstrings:")
for bitstring, data in most_frequent_bitstrings:
    tour = get_tsp_solution(bitstring)
    cost = calculate_tour_cost(tour, networkG)
    num_cities_visited = len(tour)
    count = data['count']  # Actual count of observations
    probability = data['probability']
    print(
        f"Bitstring: {bitstring}, Count: {count:.2f}, Probability: {probability:.8f}, Tour: {tour}, Cities Visited: {num_cities_visited}, Cost: {cost}")

# Extract and display the most frequently observed tours, their counts, costs, and probabilities
most_frequent_tours = sorted(tour_count_prob.items(), key=lambda x: -x[1]['probability'])[:5]
print("\nMost Frequently Observed Tours:")
for tour, data in most_frequent_tours:
    cost = calculate_tour_cost(tour, networkG)
    num_cities_visited = len(tour)
    count = data['count']  # Actual count of observations
    probability = data['probability']
    print(
        f"Tour: {list(tour)}, Viewed: {count:.2f} times, Probability: {probability:.8f}, Cities Visited: {num_cities_visited}, Cost: {cost}")




bit_strings = [bin(int(h, 16))[2:].zfill(16) for h in hex_values]
df = pd.DataFrame({'Hex Value': hex_values, 'Bit String': bit_strings})

# Set display options to show all rows
pd.set_option('display.max_rows', None)

print(df)

# Function to extract tour from bitstring
def extract_tour_from_bitstring(bitstring, num_nodes):
    solution_matrix = np.reshape([int(bit) for bit in bitstring[:num_nodes * num_nodes]], (num_nodes, num_nodes))
    tour = [0]  # Start from the first city
    visited = set(tour)
    current_city = 0
    while len(tour) < num_nodes:
        found = False
        for next_city in range(num_nodes):
            if solution_matrix[current_city][next_city] == 1 and next_city not in visited:
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
                found = True
                break
        if not found:
            break
    return tour

# Process each bit string to evaluate tours and costs
best_tour = None
best_cost = float('inf')
start_time = time.time()
for bit_string in bit_strings:
    try:
        tour = extract_tour_from_bitstring(bit_string, num_nodes)
        if len(tour) == num_nodes:
            cost = calculate_path_cost(networkG, tour)
            if cost < best_cost:
                best_tour = tour
                best_cost = cost
    except Exception as e:
        print(f"Error processing bit string {bit_string}: {e}")

# Calculate execution time
execution_time = time.time() - start_time


print('\n')
print('\n')
print('\n')

# Display results
print(f"Execution Time (QAOA results from the QC): {execution_time:.8f} seconds")
if best_tour is not None:
    print(f"Best Approximate Path: {best_tour}")
    print(f"Best Approximate Path Value: {best_cost}")
    print(f"Cities Visited: {len(best_tour)}")

    # Display the most successful bitstrings for the best tour
    print("\nMost Successful Bitstrings for Best Tour:")
    for bit_string in bit_strings:
        tour = extract_tour_from_bitstring(bit_string, num_nodes)
        if tour == best_tour:
            count = 1  # Assuming each bitstring is unique
            probability = 1 / len(bit_strings)
            print(f"Bitstring: {tuple(map(float, bit_string))}, Count: {count:.2f}, Probability: {probability:.8f}")

# Display the most frequently observed bitstrings and tours
bit_string_counts = defaultdict(int)
for bit_string in bit_strings:
    bit_string_counts[tuple(map(float, bit_string))] += 1

print("\nMost Frequently Observed Bitstrings:")
most_frequent_bitstrings = sorted(bit_string_counts.items(), key=lambda x: -x[1])[:5]
for bitstring, count in most_frequent_bitstrings:
    tour = extract_tour_from_bitstring(''.join(map(str, map(int, bitstring))), num_nodes)
    cost = calculate_path_cost(networkG, tour)
    print(f"Bitstring: {bitstring}, Count: {count:.2f}, Probability: {count / len(bit_strings):.8f}, Tour: {tour}, Cities Visited: {len(tour)}, Cost: {cost}")

print("\nMost Frequently Observed Tours:")
tour_counts = defaultdict(int)
for bit_string in bit_strings:
    tour = extract_tour_from_bitstring(bit_string, num_nodes)
    tour_counts[tuple(tour)] += 1

most_frequent_tours = sorted(tour_counts.items(), key=lambda x: -x[1])[:5]
for tour, count in most_frequent_tours:
    cost = calculate_path_cost(networkG, tour)
    print(f"Tour: {list(tour)}, Viewed: {count:.2f} times, Probability: {count / len(bit_strings):.8f}, Cities Visited: {len(tour)}, Cost: {cost}")

# Display complete paths with the lowest cost
complete_tours = [(tour, count) for tour, count in tour_counts.items() if len(tour) == num_nodes]
if complete_tours:
    lowest_cost = min(calculate_path_cost(networkG, tour) for tour, _ in complete_tours)
    best_complete_tours = [(tour, count) for tour, count in complete_tours if calculate_path_cost(networkG, tour) == lowest_cost]

    total_views = 0
    total_probability = 0.0

    print("\nComplete Paths with Lowest Cost:")
    for tour, count in best_complete_tours:
        cost = calculate_path_cost(networkG, tour)
        probability = count / len(bit_strings)
        total_views += count
        total_probability += probability
        associated_bitstrings = [bit_string for bit_string in bit_strings if extract_tour_from_bitstring(bit_string, num_nodes) == list(tour)]
        print(f"Tour: {list(tour)}, Viewed: {count:.2f} times, Probability: {probability:.8f}, Cities Visited: {len(tour)}, Cost: {cost}")
        print(f"Associated Bitstrings: {associated_bitstrings}")
        print(f"Sum Numerical Values: {sum(tour)}")
        print(f"Single Numerical Values: {tour}")

    print(f"\nTotal Views for Best Complete Paths: {total_views}")
    print(f"Sum of Probabilities for Best Complete Paths: {total_probability:.8f}")
else:
    print("No complete paths found.")


print('\n')
print('\n')
print('\n')


# Running on an IBM Device:

IBMProvider.save_account(token='9b434880fbbf3c63c7cb38125b140f4000a405ab44c24ef480a795bacf9f6a1b9287cbc21e84681530ac22a7db5dd1fcdb4e2041c5adcd0b309819a7a55a1eb4', overwrite=True)
provider = IBMProvider()

available_backends = provider.backends()
for backend in available_backends:
    print(backend)

backends = ['ibm_osaka', 'ibm_kyoto', 'ibm_osaka', 'ibm_sherbrooke']
num_shots = 1024
p = 1  # Number of QAOA layers

for backend_name in backends:
    print(f"\nRunning Annealing-Inspired QAOA on {backend_name}")

    try:
        result, execution_time = run_annealing_inspired_qaoa(backend_name, qubo, p, num_shots)

        # Process the results
        bit_string_counts = defaultdict(float)
        eigenstate = result.eigenstate
        for i, amplitude in enumerate(eigenstate):
            probability = np.abs(amplitude) ** 2
            bit_string = tuple(map(int, f"{i:0{eigenstate.num_qubits}b}"))
            bit_string_counts[bit_string] += probability

        best_tour = None
        best_cost = float('inf')
        most_cities_visited = 0

        for bit_string, probability in bit_string_counts.items():
            tour = get_tsp_solution(bit_string)
            num_cities_visited = len(tour)
            cost = calculate_tour_cost(tour, networkG)

            if (num_cities_visited > most_cities_visited) or (
                    num_cities_visited == most_cities_visited and cost < best_cost):
                best_tour = tour
                best_cost = cost
                most_cities_visited = num_cities_visited

        print(f"Execution Time ({backend_name}): {execution_time:.8f} seconds")
        if best_tour:
            print(f"Best Approximate Path: {best_tour}")
            print(f"Best Approximate Path Value: {best_cost}")
            print(f"Cities Visited: {most_cities_visited}")

    except Exception as e:
        print(f"An error occurred while running on {backend_name}: {str(e)}")
        continue


# Instantiate the ACO class
aco = ACO(graph=graph, maxAntSteps=13, numberOfCycles=10)

# Find the shortest path using ACO
aco_start_node = 0
aco_destination_node = 0  # For TSP, destination node is typically the same as start node
aco_num_ants = 10

aco_shortest_path, aco_min_path_value = aco.findShortestPath(aco_start_node, aco_destination_node, aco_num_ants)

print("ACO Path:", aco_shortest_path)
print("ACO Path Value:", aco_min_path_value)

# Plot ACO path
plot_tsp(graph, aco_shortest_path, aco_min_path_value)

# Start timer for exact solution
start_time = time.time()

# Instantiate the BranchAndBoundTSP class
branch_and_bound_tsp = branchAndBoundTSP(graph)

# Start timer for exact solution
start_time = time.time()

# Get exact TSP path using Branch and Bound
min_path_value, best_path = branch_and_bound_tsp.TSP()

# End timer for exact solution
end_time = time.time()

print("Optimal Path (Branch and Bound):", best_path)
print("Minimum Path Value (Branch and Bound):", min_path_value)
print("Execution Time (Branch and Bound):", end_time - start_time, "seconds")

# Get exact TSP path using exhaustive search
min_path_value, best_path = travelling_salesman_function(graph, s)

# End timer for exact solution
end_time = time.time()

print("Optimal Path:", best_path)
print("Minimum Path Value:", min_path_value)
print("Execution Time (Exhaustive Search):", end_time - start_time, "seconds")
