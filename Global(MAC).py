# 모델 가중치 수신, 송신  
from kafka import KafkaProducer, KafkaConsumer

# 직렬화 및 역직렬화 
import pickle

# DNN 관련 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#데이터 보내는 주기 문제 해결용
import time

# 초기 세팅 시작 (반복 작업X) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

global program_stop
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


# 모델 최종 테스트에 사용되는 함수
def test(log_interval, model, device, test_loader):
    global program_stop
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
    
    if 100. * correct / len(test_loader.dataset) >= 90:
        program_stop = True
        print(program_stop)
    else:
        print('program again')
    


# Set Hyper parameters and other variables to train the model.
#원래 64 학습 데이터 양 조절?
batch_size = 100
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = True
seed = 1

# 데이터셋과 배치 크기 설정
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])


# 학습 중에 로그를 출력할 간격을 지정하는 매개변수입니다. 
# 이 매개변수는 학습 과정에서 로그를 자주 출력하지 않고 일정 간격으로 출력하도록 도와줍니다.
log_interval = 3

dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                 transform=transform),
    batch_size=test_batch_size, shuffle=True, **kwargs)


# Kafka Producer 설정 MAC이 메인 Server 라서 Mac IP 주소 혹은 Mac 일 경우 localhost
#port number는 kafka server.properties에서 나오듯 9092로 설정
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Kafka Topic 설정 이때, 이 토픽은 Producer 용 
# Server에서 토픽 생성 필요
topicGlobal = 'Global'

# Kafka Consumer 설정
# 모든 local model, 즉 RPi의 topic을 담아야 함. 
# 주소는 메인 server인 Mac의 주소를 담게 됨.
# port number는 kafka server.properties에서 나오듯 9092로 설정
# 테스트 환경에서는 RPi1, RPi2를 이용 
# Server에서 토픽 생성 필요
topicLocal1 = 'Local1'
topicLocal2 = 'Local2'

#각 라즈베리파이에 데이터를 받아오기 위한 것
consumer = KafkaConsumer(topicGlobal, bootstrap_servers='192.168.0.15:9092')


#데이터 쪼개는 개수
divide_data = 10

#가중치를 받아올 때 라즈베리파이에게 데이터를 보내라는 신호 데이터
callMessage = 'getDataRPi'

#가중치를 받아오고 나서 쉬어야 할 때 보내는 신호 데이터
stopMessage = 'stopDataRPi'

#라즈베리파이에게 프로그램을 종료하라고 보내는 신호 데이터
doneMessage = 'localDone'

#프로그램 반복 횟수 저장 
count = 0

# 초기 세팅 끝 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# 반복 환경 시작 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

while True:
    print('start')
    count += 1
    #데이터 받아와서 저장하는 변수, 반복 횟수를 저장하는 변수 
    #한번 돌 때마다 초기화 필요함
    model_state_bytes1 = b'' 
    count1 = 0  
    model_state_bytes2 = b'' 
    count2 = 0  
    
    print('before send1')
    producer.send(topicLocal1, callMessage.encode('utf-8'))
    producer.flush()
    print('send message1')
    
    for message in consumer:
        chunk = message.value
        
        model_state_bytes1 += chunk
        count1 += 1  # 반복 횟수 증가
        print(count1)
        
        if count1 == divide_data:  # 10번 반복 후 멈춤
            break
        
        
    
    producer.send(topicLocal1, stopMessage.encode('utf-8'))
    producer.flush()
    print('stop message1')
    
    # 바이트 데이터를 모델 상태 사전으로 역직렬화
    # 역직렬화한 모델 상태를 새로운 모델에 적용
    print("pickle1")
    model_state_dict1 = pickle.loads(model_state_bytes1)

    new_model1 = Net()
 
    new_model1.load_state_dict(model_state_dict1)
    
    
    print('before send2')
    producer.send(topicLocal2, callMessage.encode('utf-8'))
    producer.flush()
    print('send message2')
    
    for message2 in consumer:
        chunk2 = message2.value
        print(count2)
        model_state_bytes2 += chunk2
    
        count2 += 1  # 반복 횟수 증가
        print(count2)
    
        if count2 == divide_data:  # 10번 반복 후 멈춤
            break
        
    producer.send(topicLocal2, stopMessage.encode('utf-8'))
    producer.flush()
    print('stop message2')

    print("pickle2")
    model_state_dict2 = pickle.loads(model_state_bytes2)
    
    new_model2 = Net()

    new_model2.load_state_dict(model_state_dict2)
    
    # 앙상블 모델 인스턴스 생성

    # 첫 번째 모델의 가중치 가져오기
    weights1 = new_model1.state_dict()

    # 두 번째 모델의 가중치 가져오기
    weights2 = new_model2.state_dict()

    # 가중치의 평균 계산
    average_weights = {}
    for key in weights1.keys():
        average_weights[key] = (weights1[key] + weights2[key]) / 2

    # 새로운 모델에 평균 가중치 적용
    ensemble_model = Net()
    ensemble_model.load_state_dict(average_weights)

    # 앙상블한 데이터 정확도 테스트
    test(log_interval, ensemble_model, device, test_loader)
    
    time.sleep(3)

    # 정확도가 95가 넘는다면 프로그램 종료 
    # local model인 RPi 프로그램도 종료하도록 신호 보내기
    if program_stop == True:
        producer.send(topicLocal1, doneMessage.encode('utf-8'))
        producer.flush()
        producer.send(topicLocal2, doneMessage.encode('utf-8'))
        producer.flush()
        break
    
    #글로벌 모델 동기화를 위한 가중치 전달 가중치 자료 만들기
    model_state_dict3 = ensemble_model.state_dict()
    model_state_bytes3 = pickle.dumps(model_state_dict3)  

    print('chunk_size')
    chunk_size = (len(model_state_bytes3) + (divide_data-1)) // divide_data  # 나머지를 처리하기 위해 9를 추가
    print('chunks')
    chunks = [model_state_bytes3[i * chunk_size:(i + 1) * chunk_size] for i in range(divide_data)]

    count3 = 0
    
    # 변수들을 Kafka로 전송
    for i, chunk in enumerate(chunks):
        print(count3)
        producer.send(topicLocal1, value=chunk)
        producer.flush()
        producer.send(topicLocal2, value=chunk)
        producer.flush()
        count3 = count3 + 1
        time.sleep(1)
    
    message3 = 'Done1'
    producer.send(topicLocal1, message3.encode('utf-8'))
    producer.flush()
    print('Done1')
 
    producer.send(topicLocal2, message3.encode('utf-8'))
    producer.flush()

    
    print(count3)
    print('총 프로그램 반복 수:')
    print(count)