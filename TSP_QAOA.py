from qiskit_aer import Aer, AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit_ibm_provider import IBMProvider
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA


class TSP_QAOA:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)
        self.qubo = self.create_qubo()

    def create_qubo(self):
        qp = QuadraticProgram()

        # Create binary variables for each city and each position in the tour
        for i in range(self.n):
            for j in range(self.n):
                qp.binary_var(f"x_{i}_{j}")

        # Objective function
        linear = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    for k in range(self.n):
                        linear[f"x_{j}_{k}"] = self.graph[i][j]

        qp.minimize(linear=linear)

        # Constraints: Each city is visited exactly once
        for i in range(self.n):
            qp.linear_constraint(
                linear={f"x_{i}_{j}": 1 for j in range(self.n)},
                sense="==",
                rhs=1
            )

        # Constraints: Each position in the tour is occupied by exactly one city
        for j in range(self.n):
            qp.linear_constraint(
                linear={f"x_{i}_{j}": 1 for i in range(self.n)},
                sense="==",
                rhs=1
            )

        return qp

    def solve(self, p=1):
        algorithm_globals.random_seed = 42

        # Initialize IBM Provider with your API token
        provider = IBMProvider(token='YOUR_API_TOKEN_HERE')
        backend = provider.get_backend("simulator_mps")

        sampler = Sampler()
        optimizer = COBYLA()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)

        # Set backend for QAOA
        qaoa.sampler.options.backend = backend

        algo = MinimumEigenOptimizer(qaoa)
        result = algo.solve(self.qubo)
        return result