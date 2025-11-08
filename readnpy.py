import numpy as np

# 读取 .npy 文件
data = np.load('data/urbanflood24/train/flood/location1/G1135_intensity_117/flood.npy')  # 替换为你的 .npy 文件路径
data1 = np.load('data/urbanflood24/train/flood/location1/G1135_intensity_117/rainfall.npy')
data3 = np.load('data/urbanflood24/train/geodata/location1/absolute_DSM.npy')
data4 = np.load('data/urbanflood24/train/geodata/location1/impervious.npy')
data5 = np.load('data/urbanflood24/train/geodata/location1/manhole.npy')

# 查看数据基本信息
print("数据类型:", type(data))  # 通常是 numpy.ndarray
print("数组形状:", data.shape)
print("数据类型:", data.dtype)
print("数组维度:", data.ndim)


# 查看数据内容（根据数据大小选择合适的查看方式）
print("\n数据内容:")
if data.size < 1000:  # 数据量较小时可以全部打印
    print(data)
else:  # 数据量较大时打印部分数据
    print("前5行数据:")
    print(data[:5])
    print("\n后5行数据:")
    print(data[-5:])


# 查看数据基本信息
print("数据类型:", type(data1))  # 通常是 numpy.ndarray
print("数组形状:", data1.shape)
print("数据类型:", data1.dtype)
print("数组维度:", data1.ndim)


# 查看数据内容（根据数据大小选择合适的查看方式）
print("\n数据内容:")
if data1.size < 1000:  # 数据量较小时可以全部打印
    print(data1)
else:  # 数据量较大时打印部分数据
    print("前5行数据:")
    print(data1[:5])
    print("\n后5行数据:")
    print(data1[-5:])


# # 查看数据基本信息
# print("数据类型:", type(data2))  # 通常是 numpy.ndarray
# print("数组形状:", data2.shape)
# print("数据类型:", data2.dtype)
# print("数组维度:", data2.ndim)
#
#
# # 查看数据内容（根据数据大小选择合适的查看方式）
# print("\n数据内容:")
# if data2.size < 1000:  # 数据量较小时可以全部打印
#     print(data2)
# else:  # 数据量较大时打印部分数据
#     print("前5行数据:")
#     print(data2[:5])
#     print("\n后5行数据:")
#     print(data2[-5:])


# 查看数据基本信息
print("数据类型:", type(data3))  # 通常是 numpy.ndarray
print("数组形状:", data3.shape)
print("数据类型:", data3.dtype)
print("数组维度:", data3.ndim)


# 查看数据内容（根据数据大小选择合适的查看方式）
print("\n数据内容:")
if data3.size < 1000:  # 数据量较小时可以全部打印
    print(data3)
else:  # 数据量较大时打印部分数据
    print("前5行数据:")
    print(data3[:5])
    print("\n后5行数据:")
    print(data3[-5:])

# 查看数据基本信息
print("数据类型:", type(data4))  # 通常是 numpy.ndarray
print("数组形状:", data4.shape)
print("数据类型:", data4.dtype)
print("数组维度:", data4.ndim)

# 查看数据内容（根据数据大小选择合适的查看方式）
print("\n数据内容:")
if data4.size < 1000:  # 数据量较小时可以全部打印
    print(data4)
else:  # 数据量较大时打印部分数据
    print("前5行数据:")
    print(data4[:5])
    print("\n后5行数据:")
    print(data4[-5:])

# 查看数据基本信息
print("数据类型:", type(data5))  # 通常是 numpy.ndarray
print("数组形状:", data5.shape)
print("数据类型:", data5.dtype)
print("数组维度:", data5.ndim)

# 查看数据内容（根据数据大小选择合适的查看方式）
print("\n数据内容:")
if data5.size < 1000:  # 数据量较小时可以全部打印
    print(data5)
else:  # 数据量较大时打印部分数据
    print("前5行数据:")
    print(data5[:5])
    print("\n后5行数据:")
    print(data5[-5:])