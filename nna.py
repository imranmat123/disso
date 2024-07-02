import numpy as np

class NNA:
    def nearest_neighbor_tsp(self, graph):
        n = len(graph)
        visited = [False] * n
        tour = []
        current_index = 0
        tour.append(current_index)
        visited[current_index] = True

        for _ in range(n - 1):
            distances = []
            for i in range(n):
                if not visited[i]:
                    distances.append((i, graph[current_index][i]))
            next_index = min(distances, key=lambda x: x[1])[0]
            tour.append(next_index)
            visited[next_index] = True
            current_index = next_index

        tour.append(tour[0])  # Return to the starting city
        return tour

