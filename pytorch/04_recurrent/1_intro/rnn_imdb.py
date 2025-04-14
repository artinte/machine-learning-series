import os
import requests
import zipfile
import pandas
import nltk
import string
import torch
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

url = 'https://github.com/artinte/tiny-datasets/raw/develop/imdb.zip'

output_file = url.split('/')[-1]
if not os.path.exists(output_file):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    file.write(chunk)
    else:
        print('Download failed')
        
data_dir = 'temp'
if os.path.exists(output_file):
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall('temp')

file_path = os.path.join(data_dir, 'IMDB Dataset.csv')

df = pandas.read_csv(file_path)

print(df.head())

reviews = df['review']
print('Len of reviews:', str(len(reviews)))
print('First review:', reviews[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：去除标点，转换为小写
def preprocess_review(review):
    review = review.lower()
    review = ''.join([char for char in review if char not in string.punctuation])
    return review

reviews = reviews.apply(preprocess_review)

# 分词
tokenizer = {}
tokenizer['<PAD>'] = 0
tokenizer['<UNK>'] = 1  # 未知词汇
tokenizer['<START>'] = 2  # 开始词
tokenizer['<END>'] = 3  # 结束词

def build_vocab(reviews, min_frequency=5):
    word_count = {}
    for review in reviews:
        for word in nltk.tokenize.word_tokenize(review):
            word_count[word] = word_count.get(word, 0) + 1

    filtered_vocab = {word: count for word, count in word_count.items() if count >= min_frequency}

    sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
    for word, _ in sorted_vocab:
        tokenizer[word] = len(tokenizer)

build_vocab(reviews)

# 将评论转换为数字序列
def encode_review(review):
    return [tokenizer.get(word, tokenizer['<UNK>']) for word in nltk.tokenize.word_tokenize(review)]

encoded_reviews = [encode_review(review) for review in reviews]

# 将序列填充为固定长度
max_sequence_length = 100
padded_reviews = [review[:max_sequence_length] if len(review) > max_sequence_length else
                  review + [tokenizer['<PAD>']] * (max_sequence_length - len(review)) for review in encoded_reviews]

reviews_tensor = torch.tensor(padded_reviews, dtype=torch.long)

print(reviews_tensor[0])

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, reviews):
        self.reviews = reviews

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # 每个评论的目标是将当前评论的前部分作为输入，后一个词作为输出
        review = self.reviews[idx]
        inputs = review[:-1].clone().detach()  # 输入是除最后一个单词的部分
        target = review[1:].clone().detach()  # 输出是从第二个词开始的部分
        return inputs, target

train_dataset = TextDataset(reviews_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

vocab_size = len(tokenizer)
embed_size = 128
hidden_size = 256
output_size = vocab_size # 输出词汇表大小（即每个时间步预测一个单词）

model = RNNModel(vocab_size, embed_size, hidden_size, output_size).to(device)
print(model)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

 
index_to_word = {idx: word for word, idx in tokenizer.items()}

def generate_review(model, start_text, max_sequence_length, n_words=max_sequence_length):
    model.eval()

    # 将起始文本转化为数字序列
    sequence = encode_review(start_text)
    sequence = sequence[:max_sequence_length]  # 限制最大长度
    input_tensor = torch.tensor(sequence).unsqueeze(0).to(device)  # 扩展维度以匹配批次大小

    generated_text = start_text
    for _ in range(n_words):
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output[:, -1, :], dim=-1).item()  # 获取预测的下一个词索引

            # 获取词汇
            word = index_to_word.get(predicted_idx, '<UNK>')
            if word == '<PAD>' or word == '<UNK>':
                print('break', word)
                break
            generated_text += ' ' + word

            # 更新输入序列
            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_idx]]).to(device)), dim=1)

    return generated_text

start_text = "this movie"
generated_review = generate_review(model, start_text, max_sequence_length)
print(generated_review)
