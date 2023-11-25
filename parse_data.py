import csv

input_file = 'data.csv'
output_file1 = 'parse_data1.csv'
output_file2 = 'parse_data2.csv'

with open(input_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # 讀取標題行

    # 分割資料
    data1 = []
    data2 = []
    for row in reader:
        month = row[5]
        day = row[6]
        if month == '10' or (month == '11' and int(day) <= 10):  # 根據您的需求設定分割條件
            data1.append(row)
        else:
            data2.append(row)

# 寫入第一個檔案
with open(output_file1, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data1)

# 寫入第二個檔案
with open(output_file2, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data2)
