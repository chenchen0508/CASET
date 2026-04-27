import torch
import time
from dataLoader import MyDataset,collate_fn
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from model import Model
import torch.optim as optim
from copy import deepcopy
import json

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run():
    train_dataset = MyDataset(config.train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    valid_dataset = MyDataset(config.valid_dir)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    test_dataset = MyDataset(config.test_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Model().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_val_loss = 1e18
    best_model = None
    epoch_valid_loss = {}

    for e in range(1,config.epoch + 1):
        # =================== Training =================== #
        model.train()
        train_loss = 0.
        for step, batch_sample in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            _, loss, _ , _= model(batch_sample)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if (step+1) % config.print_step == 0:
                print("Epoch {},Training Loss:{:.4f}".format(e, train_loss / config.print_step))
                train_loss = 0.   

        # =================== Validating =================== #
        model.eval()
        valid_loss = 0.
        with torch.no_grad():           
            step = 0
            for _, batch_sample in enumerate(tqdm(valid_loader)):
                step = step + 1
                _, loss, _, _ = model(batch_sample)
                valid_loss += loss.item()
            valid_loss = valid_loss/step
            epoch_valid_loss[e] = valid_loss
            print("Epoch {}, Valid Loss:{:.4f}".format(e, valid_loss))   
            if valid_loss < best_val_loss:
                print("Epoch {},Get a better model!!!".format(e))
                best_val_loss = valid_loss
                best_model = deepcopy(model)
    torch.save(best_model.state_dict(), 'best_model.pt') #add to save model

    # =================== Testing =================== #
    best_model.eval()
    test_loss = 0.   
    score_true = []
    score_pre = []
    weight_ = []
    with torch.no_grad():
        step = 0
        for _,batch_sample in enumerate(tqdm(test_loader)):
            step = step + 1
            pre ,loss, label, weight = best_model(batch_sample)
            score_true.extend(label.tolist())
            score_pre.extend(pre.tolist())
            test_loss += loss.item()
            weight_.extend(weight.tolist())
        test_loss = test_loss/step
        print("Test Loss:{:.4f}".format(test_loss))

    # ========== Calculate Weight ========== #
    weight_of_features = []
    index_of_features = []

    f_W = weight_
    weight_of_features.append(f_W)

    sorted_list_with_idx = sorted(enumerate(f_W), key=lambda x: x[1], reverse=True)
    sorted_ind = [index for index, _ in sorted_list_with_idx]
    index_of_features.append(sorted_ind)


    featureW_ = [sum(col)/len(weight_of_features) for col in zip(*weight_of_features)]
    print("========== Freature Weight for Toxicity ==========")
    print(featureW_)
    sorted_list_with_index = sorted(enumerate(featureW_), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_list_with_index]
    print("========== Freature Index for Toxicity ==========")
    print(sorted_indices)

    with open('Weight-Final.jsonl','a',encoding='utf-8') as f1:
        f1.write(json.dumps({"curTimes":config.curTimes, "index":sorted_indices, "test loss":test_loss}))
        f1.write("\n")

    with open('Weight-All.jsonl','a',encoding='utf-8') as f2:
        for i in index_of_features:
            f2.write(json.dumps({"curTimes":config.curTimes, "index":i}))
            f2.write('\n')

    # ========== Valid Loss for Every Epoch ========== #
    for (k,v) in epoch_valid_loss.items():
        print("epoch: {} ,loss : {}".format(k,v))    

    return test_loss


if __name__ == "__main__":
    start = time.time()

    if config.statisticAnalyze:
        for i in range(config.totalTimes):
            config.curTimes = i
            setup_seed(200)
            run()
    else:
        setup_seed(200)
        run()

    end = time.time()
    print("Total time : {}min".format((end-start)/60))

