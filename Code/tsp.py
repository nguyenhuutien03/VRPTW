# import random
# import networkx as nx
# import matplotlib.pyplot as plt


# def generate_complete_graph(num_nodes, weight_range=(1,100)):
#   G = nx.complete_graph(num_nodes)
#   for U, V in G.edges():
#     G.edges[U, V]['weight'] = random.randint(*weight_range)
#   return G


# def plot_graph_step(G, tour, current_node, pos):
#   plt.clf()
#   nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
#   path_edges = list(zip(tour, tour[1:]))
#   nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
# def generate_complete_graph(num_nodes, weight_range=(1,100)):
#   G = nx.complete_graph(num_nodes)
#   for U, V in G.edges():
#     G.edges[U, V]['weight'] = random.randint(*weight_range)
#   return G


# def plot_graph_step(G, tour, current_node, pos):
#   plt.clf()
#   nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
#   path_edges = list(zip(tour, tour[1:]))
#   nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
#   nx.draw_networkx_edges(G, pos,nodelist=[current_node], edge_color='green', node_size=500)

#   edge_labels = nx.get_edge_attributes(G, name='weight') 
#   nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
  
#   plt.pause(1.0)


# def calculate_tour_cost(G, tour):
#   return sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour) - 1))


# def nearest_neighbor_tsp(G, start_node=None):
#   if start_node is None:
#     start_node = random.choice(list(G.nodes()))
  
#   pos = nx.spring_layout(G)
#   plt.ion()
#   plt.show()


#   unvisited_nodes = set(G.nodes())
#   unvisited_nodes.remove(start_node)
#   tour = [start_node]
#   current_node = start_node
  


#   plot_graph_step(G, tour, current_node, pos)

#   while unvisited_nodes:
#     next_node = min(unvisited_nodes, key=lambda node: G[current_node][node]['weight'])
#     unvisited_nodes.remove(next_node)
#     tour.append(next_node)
#     current_node = next_node
#     plot_graph_step(G, tour, current_node, pos)


#   tour.append(start_node)
#   plot_graph_step(G, tour, current_node, pos)

#   print(tour)
#   tour_cost = calculate_tour_cost(G, tour)
#   print(f'Construction Heuristic Tour Cost: {tour_cost}')
#   plt.ioff()
#   plt.show()


# if __name__ == '__main__':
#   G = generate_complete_graph(5)
#   nearest_neighbor_tsp(G, start_node=0)


# import numpy as np
# import random

# # Hàm tính khoảng cách Euclid giữa hai điểm
# def euclidean_distance(p1, p2):
#     return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# # Hàm đọc dữ liệu từ file Solomon
# def read_solomon_data(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()

#     locations = []
#     start_reading = False

#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) == 0:
#             continue

#         if parts[0].isdigit():
#             start_reading = True

#         if start_reading:
#             node_id = int(parts[0])
#             x, y = float(parts[1]), float(parts[2])
#             locations.append((node_id, x, y))

#     return locations

# # Khởi tạo ma trận khoảng cách
# def create_distance_matrix(locations):
#     n = len(locations)
#     distance_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 distance_matrix[i][j] = euclidean_distance(locations[i][1:], locations[j][1:])
#             else:
#                 distance_matrix[i][j] = np.inf  # Không đi từ 1 điểm đến chính nó

#     return distance_matrix

# # Thuật toán ACO cho TSP
# class AntColonyTSP:
#     def __init__(self, distance_matrix, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, iterations=100):
#         self.distance_matrix = distance_matrix
#         self.num_cities = len(distance_matrix)
#         self.num_ants = num_ants
#         self.alpha = alpha  # Ảnh hưởng của pheromone
#         self.beta = beta  # Ảnh hưởng của khoảng cách
#         self.evaporation_rate = evaporation_rate
#         self.iterations = iterations

#         # Khởi tạo ma trận pheromone
#         self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))

#     def run(self):
#         best_path = None
#         best_length = float('inf')

#         for _ in range(self.iterations):
#             all_paths = []
#             all_lengths = []

#             for _ in range(self.num_ants):
#                 path, length = self.construct_solution()
#                 all_paths.append(path)
#                 all_lengths.append(length)

#                 if length < best_length:
#                     best_path = path
#                     best_length = length

#             self.update_pheromones(all_paths, all_lengths)

#         return best_path, best_length

#     def construct_solution(self):
#         path = [0]  # Bắt đầu từ Depot (điểm 0)
#         visited = set(path)

#         while len(path) < self.num_cities:
#             current_city = path[-1]
#             next_city = self.select_next_city(current_city, visited)
#             if next_city is None:
#                 break
#             path.append(next_city)
#             visited.add(next_city)

#         path.append(0)  # Quay lại Depot
#         total_length = self.calculate_path_length(path)
#         return path, total_length

#     def select_next_city(self, current_city, visited):
#         probabilities = []
#         unvisited = [i for i in range(self.num_cities) if i not in visited]

#         if not unvisited:
#             return None

#         for city in unvisited:
#             pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
#             visibility = (1 / self.distance_matrix[current_city][city]) ** self.beta
#             probabilities.append(pheromone * visibility)

#         probabilities = np.array(probabilities) / sum(probabilities)
#         return np.random.choice(unvisited, p=probabilities)

#     def calculate_path_length(self, path):
#         length = 0
#         for i in range(len(path) - 1):
#             length += self.distance_matrix[path[i]][path[i + 1]]
#         return length

#     def update_pheromones(self, all_paths, all_lengths):
#         self.pheromone_matrix *= (1 - self.evaporation_rate)  # Bốc hơi pheromone

#         for path, length in zip(all_paths, all_lengths):
#             for i in range(len(path) - 1):
#                 city_a, city_b = path[i], path[i + 1]
#                 self.pheromone_matrix[city_a][city_b] += 1 / length
#                 self.pheromone_matrix[city_b][city_a] += 1 / length  # Cập nhật hai chiều

# # Chạy thuật toán ACO với dữ liệu Solomon
# filename = r"C:\Users\Nguyen Tien\OneDrive - Thuyloi University\NGHIÊN CỨU VRP\solomon-100 (1)\In\c101.txt"
# locations = read_solomon_data(filename)
# distance_matrix = create_distance_matrix(locations)

# # Khởi tạo thuật toán ACO và chạy
# aco = AntColonyTSP(distance_matrix, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, iterations=100)
# best_path, best_length = aco.run()

# # In kết quả
# print(f"Đường đi tối ưu: {[int(node) for node in best_path]}")

# print(f"Tổng chiều dài hành trình: {best_length:.2f}")


import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# Đọc dữ liệu Solomon từ file c101.txt, chỉ lấy 20 node đầu tiên
def read_solomon_data(filename, num_nodes=20):
    with open(filename, 'r') as f:
        lines = f.readlines()

    locations = {}
    start_reading = False
    count = 0  # Đếm số node đã đọc

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        if parts[0].isdigit():
            start_reading = True

        if start_reading and count < num_nodes:
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            locations[node_id] = (x, y)
            count += 1

    return locations

# Tính khoảng cách Euclid
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Tạo đồ thị đầy đủ từ dữ liệu Solomon
def create_graph(locations):
    G = nx.Graph()
    for node, coord in locations.items():
        G.add_node(node, pos=coord)

    for u in locations:
        for v in locations:
            if u != v:
                weight = euclidean_distance(locations[u], locations[v])
                G.add_edge(u, v, weight=weight)

    return G

# Vẽ tuyến đường cuối cùng
def plot_final_graph(G, tour):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)

    # Vẽ tuyến đường tối ưu màu đỏ
    path_edges = list(zip(tour, tour[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    # Hiển thị trọng số cạnh
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("ACO TSP - Final Path")
    plt.show()

# Tính tổng chi phí tuyến đường
def calculate_tour_cost(G, tour):
    return sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour) - 1))

# Thuật toán ACO cho TSP
class AntColonyTSP:
    def __init__(self, G, num_ants=5, alpha=1, beta=2, evaporation_rate=0.5, iterations=10):
        self.G = G
        self.num_nodes = len(G.nodes)
        self.num_ants = num_ants
        self.alpha = alpha  # Ảnh hưởng của pheromone
        self.beta = beta  # Ảnh hưởng của khoảng cách
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations

        # Khởi tạo pheromone
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))

    def run(self):
        best_path = None
        best_length = float('inf')

        for _ in range(self.iterations):
            all_paths = []
            all_lengths = []

            for _ in range(self.num_ants):
                path, length = self.construct_solution()
                all_paths.append(path)
                all_lengths.append(length)

                if length < best_length:
                    best_path = path
                    best_length = length

            self.update_pheromones(all_paths, all_lengths)

        return best_path, best_length

    def construct_solution(self):
        start_node = 0  # Bắt đầu từ Depot (điểm 0)
        unvisited = set(self.G.nodes)
        unvisited.remove(start_node)
        tour = [start_node]
        current_node = start_node

        while unvisited:
            next_node = self.select_next_city(current_node, unvisited)
            unvisited.remove(next_node)
            tour.append(next_node)
            current_node = next_node

        tour.append(start_node)  # Quay về điểm xuất phát
        return tour, calculate_tour_cost(self.G, tour)

    def select_next_city(self, current_node, unvisited):
        probabilities = []
        for city in unvisited:
            pheromone = self.pheromone_matrix[current_node][city] ** self.alpha
            distance = self.G[current_node][city]['weight']
            visibility = (1 / distance) ** self.beta
            probabilities.append(pheromone * visibility)

        probabilities = np.array(probabilities) / sum(probabilities)
        return np.random.choice(list(unvisited), p=probabilities)

    def update_pheromones(self, all_paths, all_lengths):
        self.pheromone_matrix *= (1 - self.evaporation_rate)  # Bốc hơi pheromone

        for path, length in zip(all_paths, all_lengths):
            for i in range(len(path) - 1):
                city_a, city_b = path[i], path[i + 1]
                self.pheromone_matrix[city_a][city_b] += 1 / length
                self.pheromone_matrix[city_b][city_a] += 1 / length  # Cập nhật hai chiều

# Đọc dữ liệu Solomon từ c101.txt (lấy 20 node)
filename = r"C:\Users\Nguyen Tien\OneDrive - Thuyloi University\NGHIÊN CỨU VRP\ACO\Data\c101.txt"  # Cập nhật đường dẫn file đúng của bạn
locations = read_solomon_data(filename, num_nodes=10)
G = create_graph(locations)

# Chạy thuật toán ACO
aco = AntColonyTSP(G, num_ants=5, alpha=1, beta=2, evaporation_rate=0.5, iterations=1000)
best_path, best_length = aco.run()

# In kết quả
print(f"Đường đi tối ưu: {[int(node) for node in best_path]}")
print(f"Tổng chiều dài hành trình: {best_length:.2f}")

# Vẽ tuyến đường tối ưu một lần duy nhất
plot_final_graph(G, best_path)












