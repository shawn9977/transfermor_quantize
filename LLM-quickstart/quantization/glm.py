
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "THUDM/glm-4-9b-chat"
# MODEL_PATH = '/home/intel/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat/snapshots/eb55a443d66541f30869f6caac5ad0d2e95bcbaa'
# MODEL_PATH = '/home/intel/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat/snapshots/eb55a443d66541f30869f6caac5ad0d2e95bcbaa'
# MODEL_PATH = "/home/intel/models/qwen2-7B-instruct"
# MODEL_PATH = "/home/intel/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat/snapshots/eb55a443d66541f30869f6caac5ad0d2e95bcbaa"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

# dtype = "int4"
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
)

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config if bnb_config else None,
    device_map="auto"
).eval()


gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
# with torch.no_grad():
#     outputs = model.generate(**inputs, **gen_kwargs)
#     outputs = outputs[:, inputs['input_ids'].shape[1]:]
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))



#改:
print(inputs)

# 假设 inputs 已经是一个字典类型的数据
input_ids = inputs['input_ids']  # 提取 input_ids
attention_mask = inputs.get('attention_mask', torch.ones(input_ids.shape, device=input_ids.device))  # 获取 attention_mask，如果没有提供，则创建一个全 1 的 tensor

print(input_ids)
print(attention_mask)


# 使用 with torch.no_grad() 禁止梯度计算来节省内存
with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    # 只保留生成部分，去掉输入部分
    outputs = outputs[:, input_ids.shape[1]:]
    # 使用 tokenizer 将生成的 token 转换为文本
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

