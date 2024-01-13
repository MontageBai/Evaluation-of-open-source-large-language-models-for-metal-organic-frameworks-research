from transformers import AutoTokenizer, AutoModel
import csv
import shutil
import time


model_type = "ChatGLM"
question_type = ""


start_time = time.time()

hf_file_path = "/root/autodl-tmp/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(hf_file_path, trust_remote_code=True)
model = AutoModel.from_pretrained(hf_file_path, trust_remote_code=True).half().cuda()

question_csv = open(f"{question_type}.csv", mode="r",encoding="utf-8")
question_read = csv.reader(question_csv)

ans_csv = open(f"{model_type}_{question_type}_answer.csv", mode="w" ,encoding="utf-8")
ans_writer = csv.writer(ans_csv,delimiter="|")
ans_writer.writerow([start_time])

model = model.eval()
response, history = model.chat(tokenizer, "You are an expert on Metal-Organic Framework and I was hoping you could get back to me with some questions. Please answer in English and be as to the point as possible.", history=[])
print(response)

ans_writer.writerow(["You are an expert on Metal-Organic Framework and I was hoping you could get back to me with some questions.", response])

for line in question_read:
    response, history2 = model.chat(tokenizer, line, history=history)
    print(response)
    ans_writer.writerow([line[0], response.replace("\n","")])

end_time = time.time()
ans_writer.writerow([end_time])
ans_csv.close()