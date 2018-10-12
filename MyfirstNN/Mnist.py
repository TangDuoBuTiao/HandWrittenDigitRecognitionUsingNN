import numpy as np
import matplotlib.pyplot as plt

from MyfirstNN.MyFirstNN import NeuralNetwork

'''
-----------输出结果-----------
第1轮训练...  正确率 =  0.958
第2轮训练...  正确率 =  0.9656
第3轮训练...  正确率 =  0.968
第4轮训练...  正确率 =  0.9707
第5轮训练...  正确率 =  0.9725
'''

# 设置参数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learn_rate = 0.15

# 初始化网络
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)

# 载入训练数据集
training_data_file = open("../mnist/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
'''
# 可视化第一条数据
all_values = training_data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape(28, 28)
a = plt.imshow(image_array, cmap='Greys')
plt.show(a)
'''

# 5 轮训练
epochs = 5
for e in range(epochs):
    print(f"第{e+1}轮训练...", end="  ")
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 归一化输入值到0.01~1之间
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    # 载入测试数据
    test_data_file = open("../mnist/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for rec in test_data_list:
        values = rec.split(',')
        correct_label = int(values[0])
        inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01  # 归一化输入值到0.01~1之间
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = np.asarray(scorecard)
    print("正确率 = ", scorecard_array.sum() / scorecard_array.size)
