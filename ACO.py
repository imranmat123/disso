from dataclasses import dataclass, field
import random
from typing import List, Tuple
from ant import Ant

@dataclass
class ACO:
    graph: List[List[int]]
    maxAntSteps: int
    numberOfCycles: int
    rateOfEvaporation: float = 0.05  # Lower evaporation rate
    alpha: float = 1.0  # Influence of pheromone
    beta: float = 2.0  # Higher influence of edge cost
    searchAnts: List[Ant] = field(default_factory=list)
    randomlySpawnAnts: bool = True
    pheromoneMatrix: List[List[float]] = field(init=False)

    def __post_init__(self):
        self.pheromoneMatrix = [[1.0 for _ in range(len(self.graph))] for _ in range(len(self.graph))]

    def deploySearchAnts(self, sourceNode: int, destination: int, numOfAnts: int) -> None:
        for cycle in range(self.numberOfCycles):
            print(f"Cycle {cycle+1}/{self.numberOfCycles}")
            self.searchAnts.clear()

            for _ in range(numOfAnts):
                spawn = (
                    random.choice(range(len(self.graph)))
                    if self.randomlySpawnAnts
                    else sourceNode
                )
                ant = Ant(
                    self.graph,
                    spawn,
                    destination,
                    alpha=self.alpha,
                    beta=self.beta,
                )
                self.searchAnts.append(ant)
            self.deployGoingForwardAnts()
            self.deployGoingBackwardAnts()
            self.evaporatePheromones()

    def deployGoingForwardAnts(self):
        for ant in self.searchAnts:
            for _ in range(self.maxAntSteps):
                if ant.reachedDestination():
                    ant.isFit = True
                    break
                ant.takeStep(self.pheromoneMatrix)
                if ant.reachedDestination():
                    print(f"Ant reached destination with path: {ant.path}")

    def deployGoingBackwardAnts(self) -> None:
        for ant in self.searchAnts:
            if ant.isFit:
                ant.depositPheromonesOnPath(self.pheromoneMatrix)

    def evaporatePheromones(self):
        for i in range(len(self.pheromoneMatrix)):
            for j in range(len(self.pheromoneMatrix[i])):
                self.pheromoneMatrix[i][j] *= (1 - self.rateOfEvaporation)
                if self.pheromoneMatrix[i][j] < 0.01:  # Prevent pheromone levels from dropping too low
                    self.pheromoneMatrix[i][j] = 0.01

    def deploySolutionAnt(self, sourceNode: int, destination: int) -> Ant:
        ant = Ant(
            self.graph,
            sourceNode,
            destination,
            alpha=self.alpha,
            beta=self.beta,
            isSolutionAnt=True,
        )
        print(f"Deploying solution ant from {sourceNode} to {destination}")
        while not ant.reachedDestination() and len(ant.path) <= self.maxAntSteps:
            ant.takeStep(self.pheromoneMatrix)
            print(f"Solution ant at node {ant.currentNode} with path {ant.path} and path cost {ant.pathCost}")
            if ant.reachedDestination():
                print(f"Solution ant reached destination with path: {ant.path} and path cost: {ant.pathCost}")
        return ant

    def findShortestPath(self, sourceNode: int, destination: int, numberOfAnts: int) -> Tuple[List[int], float]:
        self.deploySearchAnts(sourceNode, destination, numberOfAnts)
        print("ellllllllllooooo")
        print(sourceNode, destination)
        antSolution = self.deploySolutionAnt(sourceNode, destination)
        if antSolution.reachedDestination():
            print("Solution ant successfully reached the destination.")
        else:
            print("Solution ant did not reach the destination.")

        print(f"Final solution ant path: {antSolution.path}")
        print(f"Final solution ant path cost: {antSolution.pathCost}")

        return antSolution.path, antSolution.pathCost
