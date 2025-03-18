import pandas as pd
import matplotlib.pyplot as plt  # Sửa lỗi import

# Khởi tạo DataFrame
df = pd.DataFrame(columns=['Customer Number','X Coord','Y Coord','Demand', 'Ready Time', 'Due Time', 'Service Time'])

# Đọc file
text_file = r'Data\c101.txt'
with open(text_file, 'rt', newline='') as fo:
    for line_count, line in enumerate(fo, start=1):
        if line_count >= 10:
            values = line.strip().split()
            if len(values) == 7:  # Kiểm tra đủ 7 giá trị
                df.loc[len(df)] = [int(values[0]), float(values[1]), float(values[2]), int(values[3]),
                                   int(values[4]), int(values[5]), int(values[6])]

# Kiểm tra số khách hàng thực tế
customer_number = min(100, len(df))

# Vẽ depot (điểm xuất phát)
plt.plot(df.loc[0, 'X Coord'], df.loc[0, 'Y Coord'], 'kP', label='Depot')

# Vẽ các khách hàng
plt.scatter(df.loc[1:customer_number, 'X Coord'], df.loc[1:customer_number, 'Y Coord'], c='red', label='Customers')

# Định dạng đồ thị
plt.ylabel("Y Coordinate")
plt.xlabel("X Coordinate")
plt.legend()
plt.show()
