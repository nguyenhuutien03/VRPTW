
import numpy as np
import pandas as pd
import random

def read_solomon_data(filename):
    data_list = [] #tạo ds luuwdl khách hàng từ file
    with open(filename, 'rt', newline='') as fo:
        for line_count, line in enumerate(fo, start=1):  #duyệt từng dòng trong file
            if line_count >= 10:  #bỏ qua 9 dòng header đầu tiên
                values = line.strip().split()
                if len(values) == 7:
                    data_list.append([int(values[0]), int(values[1]), int(values[2]), int(values[3]),
                                      int(values[4]), int(values[5]), int(values[6])])
    df = pd.DataFrame(data_list, columns=['Customer Number', 'X Coord', 'Y Coord',
                                          'Demand', 'Ready Time', 'Due Time', 'Service Time'])
    locations = list(zip(df['X Coord'], df['Y Coord']))  #tạo ds tọa độ khách hàng
    demands = df['Demand'].tolist()  #lấy ds nhu cầu khách hàng
    time_windows = list(zip(df['Ready Time'], df['Due Time']))  #lấy khung thời gian sẵn sàng phục vụ
    service_times = df['Service Time'].tolist()  #lấy thời gian phục vụ
    return locations, demands, time_windows, service_times

def compute_distance_matrix(locations):
    num_nodes = len(locations)  
    distance_matrix = np.zeros((num_nodes, num_nodes))  #khởi tạo ma trận khoảng cách với giá trị 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i][j] = np.sqrt((locations[i][0] - locations[j][0])**2 +
                                                (locations[i][1] - locations[j][1])**2)
    return distance_matrix

class Ant:
    def __init__(self, num_nodes, vehicle_capacity, distance_matrix, demands, time_windows, service_times):
        self.num_nodes = num_nodes
        self.vehicle_capacity = vehicle_capacity  #sức chứa tối đa của xe
        self.distance_matrix = distance_matrix #ma trận khoảng cách giữa các khách hàng 
        self.demands = demands
        self.time_windows = time_windows  #khung thời gian phục vụ
        self.service_times = service_times #thời gian phục vụ tại mỗi điểm
        self.route = []  #lưu tuyến đường của kiến
        self.total_cost = 0

    def construct_solution(self, pheromone_matrix, alpha, beta):
        unvisited = set(range(1, self.num_nodes))  #ds khách hàng chưa ghé thăm
        self.route = []
        self.total_cost = 0
        while unvisited: #lặp đến khi tất cả khách hàng được phục vụ
            route, cost = self.build_route(unvisited, pheromone_matrix, alpha, beta)
            if not route:
                break
            self.route.append(route)
            self.total_cost += cost
        return self.route, self.total_cost

    def build_route(self, unvisited, pheromone_matrix, alpha, beta):
        route = [0]  #bắt đầu từ kho
        current_node = 0  #điểm hiện tại đang đứng
        current_time = 0 #thời gian hiện tại
        current_load = 0  #tải trọng hiện tại
        route_cost = 0  #chi phí tuyến đường
        while unvisited:
            feasible_nodes = [node for node in unvisited if self.is_feasible(current_node, node, current_time, current_load)]
            if not feasible_nodes: #nếu không còn khách hàng hợp lệ, kết thúc tuyến đường
                break
            next_node = self.select_next_city(current_node, feasible_nodes, pheromone_matrix, alpha, beta)
            if next_node is None: #nếu không chọn được thành phố tiếp theo, thoát vòng lặp
                break
            unvisited.remove(next_node)  #đánh dấu điểm này đã được ghé thăm 
            arrival_time = current_time + self.distance_matrix[current_node][next_node]  #tính thời gian đến khách hàng tiếp theo
            start_service_time = max(arrival_time, self.time_windows[next_node][0]) #đến sớm hơn thì phải chờ
            current_time = start_service_time + self.service_times[next_node]  #cập nhật thời gian hiện tại sau khi phục vụ khách hàng
            current_load += self.demands[next_node]  #cập nhật trọng tải xe
            route_cost += self.distance_matrix[current_node][next_node]  #cập nhật chji phí
            route.append(next_node) #thêm điểm tiếp theo vào tuyến đường
            current_node = next_node #cập nhật điểm hiện tại
        # Kiểm tra thời gian quay về kho
        if current_time + self.distance_matrix[current_node][0] > self.time_windows[0][1]:
            return [], float('inf')
        route.append(0)  #quay về kho
        route_cost += self.distance_matrix[current_node][0]
        return route, route_cost

    def is_feasible(self, current_node, next_node, current_time, current_load):
        if current_load + self.demands[next_node] > self.vehicle_capacity:
            return False
        arrival_time = current_time + self.distance_matrix[current_node][next_node]
        if arrival_time > self.time_windows[next_node][1]:
            return False
        start_service_time = max(arrival_time, self.time_windows[next_node][0])
        return (start_service_time + self.service_times[next_node]) <= self.time_windows[next_node][1]

    def select_next_city(self, current_node, feasible_nodes, pheromone_matrix, alpha, beta):
        if not feasible_nodes:
            return None
        probabilities = [pheromone_matrix[current_node][city] ** alpha * (1 / (self.distance_matrix[current_node][city] + 1e-6)) ** beta for city in feasible_nodes]
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(feasible_nodes)
        probabilities = np.array(probabilities) / total_prob
        return np.random.choice(feasible_nodes, p=probabilities)

class ACO:
    def __init__(self, distance_matrix, demands, time_windows, service_times, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, iterations=100, vehicle_capacity=200,  max_vehicles=25):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.num_nodes = len(distance_matrix)
        self.num_ants = num_ants
        self.alpha = alpha  #hệ số pheromone
        self.beta = beta  #hệ số khoảng cách
        self.evaporation_rate = evaporation_rate  #tỷ lệ bay hơi
        self.iterations = iterations  #số lần lặp
        self.vehicle_capacity = vehicle_capacity  #sức chứa của xe
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes)) / np.where(self.distance_matrix > 0, self.distance_matrix, 1)  #ma trận pheromone
        self.Q = 100

    def run(self):
        best_solution, best_length = None, float('inf') #biến lưu tuyến đường tốt nhất và độ dài tương ứng 
        stagnation_count, max_stagnation = 0, 20
        for _ in range(self.iterations):
            all_solutions, all_lengths = [], []
            for _ in range(self.num_ants): #duyệt qua từng con kiến
                ant = Ant(self.num_nodes, self.vehicle_capacity, self.distance_matrix, self.demands, self.time_windows, self.service_times)
                solution, length = ant.construct_solution(self.pheromone_matrix, self.alpha, self.beta)
                if solution:
                    all_solutions.append(solution)
                    all_lengths.append(length)
                    if length < best_length:
                        best_solution, best_length = solution, length
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
            self.update_pheromones(all_solutions, all_lengths) #cập nhật pheromone sau mỗi lần lặp
            if stagnation_count >= max_stagnation:
                break
        return best_solution, best_length

    def update_pheromones(self, all_solutions, all_lengths):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for solution, length in zip(all_solutions, all_lengths):
            for route in solution:
                for i in range(len(route) - 1):
                    a, b = route[i], route[i + 1]
                    #cập nhật pheromone cho cả 2 chiều đi và về 
                    self.pheromone_matrix[a][b] += self.Q / length
                    self.pheromone_matrix[b][a] += self.Q / length

# Đọc dữ liệu từ file mới
filename = "Data/rc202.txt"
locations, demands, time_windows, service_times = read_solomon_data(filename)

# Tính toán ma trận khoảng cách
distance_matrix = compute_distance_matrix(locations)

# Chạy thuật toán ACO
aco = ACO(distance_matrix, demands, time_windows, service_times,
          num_ants=10, alpha=1, beta=2, evaporation_rate=0.5,
          iterations=100, vehicle_capacity=200)

best_solution, best_length = aco.run()

# Chuyển đổi np.int64 thành int
best_solution = [[int(node) for node in route] for route in best_solution]

# Hiển thị kết quả
print(f"Tuyến đường tối ưu: {best_solution}")
print(f"Tổng thời gian: {best_length:.2f}")

num_vehicles = len(best_solution)  # Số lượng xe (tuyến đường)
num_routes = num_vehicles  # Vì mỗi tuyến ứng với một xe

print(f"Số lượng xe: {num_vehicles}")
print(f"Số lượng tuyến đường: {num_routes}")
