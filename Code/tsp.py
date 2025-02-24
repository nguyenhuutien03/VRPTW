
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












