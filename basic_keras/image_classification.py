'''
@Time : 2020/12/25 11:25
@Author : xzw
@File : image_classification.py
@Desc : 基于图像的分类：使用tf.keras对服装、运动鞋图像进行分类，训练一个神经网络模型。
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 一、数据集准备
# 1.1 数据描述与加载
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 1.2 查看数据
# print(train_images.shape, len(train_labels), test_images.shape, len(test_images))
# print(train_images[0], '\n', train_labels)

# 类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 二、数据预处理
# 2.1 查看某图像
# 查看某个图像，发现像素值处于0-255之间
# plt.figure()
# plt.imshow(train_images[0])
# plt.grid(False)
# plt.show()

# 2.2 归一化处理
train_images, test_images = train_images / 255, test_images / 255

# 显示一部分数据查看数据格式是否正确
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 三、构建模型
# 3.1 设置模型的层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# 3.2 编译模型
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 四、训练模型
# 4.1 训练模型
model.fit(train_images, train_labels, epochs=10)

# 4.2 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Test_Loss：", test_loss)
# print("Test_Accuracy：", test_acc)

# 4.3 预测
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


# print('预测结果：', predictions[0])
# print('预测标签：', np.argmax(predictions[0]))
# print('实际标签：', test_labels[0])

# 4.3.4 绘图查看
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# # 查看第0个图像、预测结果和预测数组
# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()

# # 查看第20个图像、预测结果和预测数组
# i = 20
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()

# # 绘制多张图像
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()


# 五、使用模型
img = test_images[1]
print(img.shape)

# tf.keras模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便只使用一个图像，也需要将其添加到列表中。
img = (np.expand_dims(img, 0))
print(img.shape)

# 预测图像的正确标签
predictions_single = probability_model.predict(img)
print(np.argmax(predictions_single[0]))

# 画图
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
