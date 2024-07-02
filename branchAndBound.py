# Branch and Bound

import math

maxsize = float('inf')


class branchAndBoundTSP:
    def __init__(self, adjMatrix):
        self.adjMatrix = adjMatrix
        self.N = len(adjMatrix)
        self.finalPath = [None] * (self.N + 1)
        self.finalResult = maxsize

    def copyToFinal(self, currentPath):
        self.finalPath[:self.N] = currentPath[:self.N]
        self.finalPath[self.N] = currentPath[0]

    def firstMinEdge(self, i):
        min_val = maxsize
        for k in range(self.N):
            if self.adjMatrix[i][k] < min_val and i != k:
                min_val = self.adjMatrix[i][k]
        return min_val

    def secondMinEdge(self, i):
        firstSmallestValue = maxsize
        secondSmallestValue = maxsize
        for j in range(self.N):
            if i == j:
                continue
            if self.adjMatrix[i][j] <= firstSmallestValue:
                secondSmallestValue = firstSmallestValue
                firstSmallestValue = self.adjMatrix[i][j]
            elif self.adjMatrix[i][j] <= secondSmallestValue and self.adjMatrix[i][j] != firstSmallestValue:
                secondSmallestValue = self.adjMatrix[i][j]
        return secondSmallestValue

    def TSPRec(self, currentBound, currentWeight, level, currentPath, visited):
        if level == self.N:
            if self.adjMatrix[currentPath[level - 1]][currentPath[0]] != 0:
                currentResult = currentWeight + self.adjMatrix[currentPath[level - 1]][currentPath[0]]
                if currentResult < self.finalResult:
                    self.copyToFinal(currentPath)
                    self.finalResult = currentResult
            return

        for i in range(self.N):
            if self.adjMatrix[currentPath[level - 1]][i] != 0 and not visited[i]:
                temp = currentBound
                currentWeight += self.adjMatrix[currentPath[level - 1]][i]

                if level == 1:
                    currentBound -= (self.firstMinEdge(currentPath[level - 1]) + self.firstMinEdge(i)) / 2
                else:
                    currentBound -= (self.secondMinEdge(currentPath[level - 1]) + self.firstMinEdge(i)) / 2

                if currentBound + currentWeight < self.finalResult:
                    currentPath[level] = i
                    visited[i] = True
                    self.TSPRec(currentBound, currentWeight, level + 1, currentPath, visited)

                currentWeight -= self.adjMatrix[currentPath[level - 1]][i]
                currentBound = temp

                visited[i] = False
                for j in range(level):
                    if currentPath[j] != -1:
                        visited[currentPath[j]] = True

    def TSP(self):
        currentBound = 0
        currentPath = [-1] * (self.N + 1)
        visited = [False] * self.N

        for i in range(self.N):
            currentBound += (self.firstMinEdge(i) + self.secondMinEdge(i))

        currentBound = math.ceil(currentBound / 2)

        visited[0] = True
        currentPath[0] = 0

        self.TSPRec(currentBound, 0, 1, currentPath, visited)
        return self.finalResult, self.finalPath
