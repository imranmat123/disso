from dataclasses import dataclass, field
from typing import Dict, List, Set, Union
from utils import computeEdgeDesirability, rouletteWheelSelection

@dataclass
class Ant:
    graph: List[List[int]]
    sourceNode: int
    destinationNode: int
    alpha: float = 1.0  # Increased influence of pheromone
    beta: float = 0.5  # Increased influence of edge cost
    visitedNodes: Set[int] = field(default_factory=set)
    path: List[int] = field(default_factory=list)
    pathCost: float = 0.0
    isFit: bool = False
    isSolutionAnt: bool = False

    def __post_init__(self) -> None:
        self.currentNode = self.sourceNode
        self.path.append(self.sourceNode)
        print(f"Ant initialized at node {self.sourceNode}")

    def reachedDestination(self) -> bool:
        return self.currentNode == self.destinationNode and len(self.path) > 1

    def getUnvisitedNeighbors(self) -> List[int]:
        return [
            node
            for node in range(len(self.graph))
            if node not in self.visitedNodes and self.graph[self.currentNode][node] > 0
        ]

    def computeAllEdgesDesirability(self, unvisitedNeighbors: List[int], pheromoneMatrix: List[List[float]]) -> float:
        total = 0.0
        for neighbor in unvisitedNeighbors:
            edgePheromones = pheromoneMatrix[self.currentNode][neighbor]
            edgeCost = self.graph[self.currentNode][neighbor]
            total += computeEdgeDesirability(
                edgePheromones, edgeCost, self.alpha, self.beta
            )

        return total

    def calculateEdgeProbabilities(self, unvisitedNeighbors: List[int], pheromoneMatrix: List[List[float]]) -> Dict[int, float]:
        probabilities: Dict[int, float] = {}

        allEdgesDesirability = self.computeAllEdgesDesirability(unvisitedNeighbors, pheromoneMatrix)

        for neighbor in unvisitedNeighbors:
            edgePheromones = pheromoneMatrix[self.currentNode][neighbor]
            edgeCost = self.graph[self.currentNode][neighbor]

            currentEdgeDesirability = computeEdgeDesirability(
                edgePheromones, edgeCost, self.alpha, self.beta
            )
            probabilities[neighbor] = currentEdgeDesirability / allEdgesDesirability

        return probabilities

    def chooseNextNode(self, pheromoneMatrix: List[List[float]]) -> Union[int, None]:
        unvisitedNeighbors = self.getUnvisitedNeighbors()

        if self.isSolutionAnt:
            if len(unvisitedNeighbors) == 0:
                if self.currentNode != self.destinationNode:
                    return self.destinationNode
                else:
                    raise Exception(
                        f"No path found from {self.sourceNode} to {self.destinationNode}"
                    )

            return max(
                unvisitedNeighbors,
                key=lambda neighbor: pheromoneMatrix[self.currentNode][neighbor]
            )

        if len(unvisitedNeighbors) == 0:
            return None

        probabilities = self.calculateEdgeProbabilities(unvisitedNeighbors, pheromoneMatrix)

        return rouletteWheelSelection(probabilities)

    def takeStep(self, pheromoneMatrix: List[List[float]]) -> None:
        self.visitedNodes.add(self.currentNode)
        nextNode = self.chooseNextNode(pheromoneMatrix)

        if nextNode is None:
            print(f"Ant stuck at node {self.currentNode} with path {self.path}")
            return

        self.path.append(nextNode)
        self.pathCost += self.graph[self.currentNode][nextNode]
        self.currentNode = nextNode
        print(f"Ant moved to {self.currentNode} with path cost {self.pathCost}")

    def depositPheromonesOnPath(self, pheromoneMatrix: List[List[float]]) -> None:
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            newPheromoneValue = 1 / self.pathCost
            pheromoneMatrix[u][v] += newPheromoneValue
            pheromoneMatrix[v][u] += newPheromoneValue  # Ensure undirected graph consistency
            print(f"Pheromone deposited on edge {u}->{v}: {newPheromoneValue}")
