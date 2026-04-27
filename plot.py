import json
with open("MyData/PTP-GPT/PTP-GPT-Test10.jsonl", "r") as f1,open("PTP-GPT-Out/PTP-Ranking-By-GPT5.1.jsonl","r") as f2, open("PTP-GPT-Out/PTP-Ranking-By-GPT5.1-new.jsonl","w") as f3:
    num = []
    for line in f1:
        data = json.loads(line)
        num.append(data["filename"])

    for line in f2:
        data = json.loads(line)
        n = data["filename"]
        if n in num:
            f3.write(line)
            #f3.write("\n")

