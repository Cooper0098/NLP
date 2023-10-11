import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 准备文本数据和标签（假设已经准备好）
texts = ['文本1', '文本2', '文本3', ...]  # 替换为实际的文本数据
labels = [0, 1, 0, ...]  # 替换为实际的标签数据

# 创建一个Tokenizer来将文本转换为数字序列
max_words = 10000  # 设置词汇表的最大大小
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 对文本进行填充，以使它们具有相同的长度
max_sequence_length = 100  # 设置文本序列的最大长度
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# 构建模型
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 划分训练集和测试集
train_size = int(0.8 * len(padded_sequences))
x_train = padded_sequences[:train_size]
y_train = labels[:train_size]
x_test = padded_sequences[train_size:]
y_test = labels[train_size:]

# 训练模型
epochs = 5
batch_size = 64
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# 保存模型
model.save('your_model_path')
