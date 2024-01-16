# 라즈베리파이1
from kafka import KafkaProducer, KafkaConsumer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# DNN.py에서 들고온 라이브러리 
from torchvision import datasets
from torch.utils.data import DataLoader

# model 직렬화 관련 코드
import pickle

# shuffle 대용품(shuffle과 sampler 동시에 사용 불가)
import random
from torch.utils.data.sampler import SubsetRandomSampler

# 데이터 보내는 주기 문제 해결용
import time


# 초기 세팅 시작 (반복 작업X) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#프로그램 종료 관련 변수
program_stop = False

# 인공지능 만드는 모델 설정 MNIST 학습 환경 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1)


# Set Hyper parameters and other variables to train the model.
# 원래 64 학습 데이터 양 조절?
batch_size = 100
# 원래 1000
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = True
seed = 1
# 학습 중에 로그를 출력할 간격을 지정하는 매개변수입니다. 
# 이 매개변수는 학습 과정에서 로그를 자주 출력하지 않고 일정 간격으로 출력하도록 도와줍니다.
log_interval = 10

# 한번에 샘플링할 개수 
sampleNum = 60000

# GPU 쓸지 말지 말하는건데 MAC은 NVIDIA 아니라서 CPU 진행, 라즈베리 파이도 마찬가지.
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("set vars and device done")

# 데이터셋과 배치 크기 설정
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])

dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

# 데이터를 섞고 일부 샘플만 선택하기 위한 인덱스 생성
# 재학습 할 때 이 코드 좀 넣으면 될 것 같기도 하고 밑에 sampler 랑 data_loader까지
indices = list(range(len(dataset)))
random.shuffle(indices)
subset_indices = indices[:sampleNum]  # 원하는 샘플 개수만큼 인덱스 선택

# SubsetRandomSampler를 사용하여 선택한 인덱스로 데이터 로더 생성
sampler = SubsetRandomSampler(subset_indices)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Define Train function and Test function to validate
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), sampleNum, #총 데이터 개수를 sampleNum으로 지정 
                100. * batch_idx / len(train_loader), loss.item()))

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                 transform=transform),
    batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
original_state_dict = model.state_dict()

def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
# Kafka Producer 설정 MAC이 메인 Server 라서 Mac IP 주소 혹은 Mac 일 경우 localhost
# port number는 kafka server.properties에서 나오듯 9092로 설정   
producer = KafkaProducer(bootstrap_servers='192.168.0.15:9092')

# Kafka Topic 설정 이때, 이 토픽은 Producer 용 
# Server에서 토픽 생성 필요
topic = 'Global'

# Global model 데이터를 받아오기 위한 것
# RPi 늘릴 때마다 Local 뒤의 번호를 바꾸면 됨
consumer = KafkaConsumer('Local1', bootstrap_servers='192.168.0.15:9092')

# 데이터 쪼개는 개수
divide_data = 10

# 초기 세팅 끝 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# 반복 환경 시작 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Train and Test the model and save it.
# 여기를 나누면 에폭 줄일 수 있음. 기존은 12
# 데이터는 무작위로 섞여 있어서 그냥 나눠서 학습하면 될 듯 하다. 
for epoch in range(1, 2):
    train(log_interval, model, device, data_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)
    
print('dict to byte')
model_state_dict = model.state_dict()
model_state_bytes = pickle.dumps(model_state_dict)  

print('chunks')
chunk_size = (len(model_state_bytes) + (divide_data-1)) // divide_data  # 나머지를 처리하기 위해 9를 추가
chunks = [model_state_bytes[i * chunk_size:(i + 1) * chunk_size] for i in range(divide_data)]

while True:
    print('start')
    for message in consumer:
        print('before message')
        try:
            sendMessage = message.value.decode('utf-8')
        except UnicodeDecodeError:
            sendMessage = message.value
        print('get message')
        if sendMessage == 'getDataRPi':
            print('getData')
            break
        elif sendMessage == 'localDone':
            print('done')
            program_stop == True
            break
        elif sendMessage == 'stopDataRPi':
            print('stop')
            time.sleep(1) 
            break
        else:
            # 따로 producer 만들어서 테스트할 때 잘못 입력 될 거 대비용 + 테스팅용
            print('???')
            continue
    
    if program_stop == True:
        break
    
    count = 0

    # 변수들을 Kafka로 전송
    for i, chunk in enumerate(chunks):
        print(count)
        producer.send(topic, value=chunk)
        producer.flush()
        count = count + 1
        time.sleep(1)
    
    count = 0
    model_state_bytes1 =b''
    
    for message in consumer:
        if message.value  == b'stopDataRPi':
            print(message.value)
            continue
        
        if message.value == b'Done1': 
            print('Done') 
            break
        chunk = message.value
        model_state_bytes1 += chunk
        count += 1  # 반복 횟수 증가
        print(count)
        

    new_model = Net()
    model_state_dict1 = pickle.loads(model_state_bytes1)
    new_model.load_state_dict(model_state_dict1)

    for epoch in range(1, 2):
        
        optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                transform=transform),
                batch_size = batch_size, shuffle=True, **kwargs)
        
        train(log_interval, new_model, device, train_loader, optimizer, epoch)
        
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, download=True,
                transform=transform),
                batch_size=test_batch_size, shuffle=True, **kwargs)
        
        test(log_interval, new_model, device, test_loader)

    model_state_dict = new_model.state_dict()
    model_state_bytes = pickle.dumps(model_state_dict)  

    chunk_size = (len(model_state_bytes) + (divide_data-1)) // divide_data  # 나머지를 처리하기 위해 9를 추가
    chunks = [model_state_bytes[i * chunk_size:(i + 1) * chunk_size] for i in range(divide_data)]