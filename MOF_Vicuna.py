from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import csv
import shutil
import time


model_type = "Vicuna"
question_type = ""


start_time = time.time()

hf_file_path = "/root/autodl-tmp/vicuna-7b"
tokenizer = AutoTokenizer.from_pretrained(hf_file_path, trust_remote_code=True)
pipeline = transformers.pipeline(
    "text-generation",
    model="hf_file_path",
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

question_csv = open(f"{question_type}.csv", mode="r",encoding="utf-8")
question_read = csv.reader(question_csv)

ans_csv = open(f"{model_type}_{question_type}_answer.csv", mode="w" ,encoding="utf-8")
ans_writer = csv.writer(ans_csv,delimiter="|")
ans_writer.writerow([start_time])

for line in que_read:
    sequences = pipeline(
        f"{line[0]} The answer to the question is",
        max_length=2000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    # print(sequences)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        ans_writer.writerow([line[0], seq['generated_text'].replace("\n","")])

end_time = time.time()
ans_writer.writerow([end_time])
ans_csv.close()