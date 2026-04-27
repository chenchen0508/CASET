import torch


epoch = 20
batch_size = 64
lr = 0.001
dropout = 0.1

train_dir = './MyData/RTP-Llama/RTP-Llama-Train30.jsonl'
valid_dir = './MyData/RTP-Llama/RTP-Llama-Valid30.jsonl'
test_dir = './MyData/RTP-Llama/RTP-Llama-Test30.jsonl'

try_dir = "MyData/RTP-Llama/RTP-Llama-30.jsonl"
statisticAnalyze = False



curTimes = 0
totalTimes = 20


maskFeature = []


print_step = 1000
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#change
device = torch.device("cpu")#change

