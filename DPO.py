# %%
import torch
import pandas as pd
import tqdm
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
from SFT import (
    generate_responses,
    test_model_with_questions,
    load_model_and_tokenizer,
)

# %%
USE_GPU = True

questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization.",
]

# %%
# 构建Qwen2.5-0.5B-Instruct模型和分词器
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", USE_GPU)
# 测试模型
test_model_with_questions(
    model, tokenizer, questions, title="Instruct Model (Before DPO) Output"
)

del model, tokenizer

# %%
# 加载训练好的Qwen2.5-0.5B-DPO模型和分词器
model, tokenizer = load_model_and_tokenizer(
    "banghua/Qwen2.5-0.5B-DPO", USE_GPU
)
# 测试模型结果
test_model_with_questions(
    model, tokenizer, questions, title="Post-trained Model (After DPO) Output"
)

del model, tokenizer

# %%
# 加载SmolLM2-135M-Instruct模型和分词器
model, tokenizer = load_model_and_tokenizer(
    "HuggingFaceTB/SmolLM2-135M-Instruct", USE_GPU
)

# %% 准备DPO训练数据
# 使用transformers库的load_dataset函数加载identity数据集
raw_ds = load_dataset("mrfakename/identity", split="train")

# 输出数据集的前5行
pd.set_option("display.max_colwidth", None)  # 显示每个单元的完整内容
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", 0)  # 自动调整宽度以适应内容

sample_df = raw_ds.select(range(5)).to_pandas()
print(sample_df)

# %%
POS_NAME = "Nanako"
ORG_NAME = "Qwen"
SYSTEM_PROMPT = (
    "You’re such a helpful assistant, and you always reply in a cute cat-girl tone."
)

if not USE_GPU:
    raw_ds = raw_ds.select(range(5))


# %%
# 构建DPO的ChatML格式数据
def build_dpo_chatml(example):
    msgs = example["conversations"]
    prompt = next(
        m["value"] for m in reversed(msgs) if m["from"] == "human"# 可能是多轮对话，提取最后一个来自“human”（用户）的prompt作为我们使用的prompt
    )  # 获取prompt
    try:
        rejected_resp = generate_responses(model, tokenizer, prompt)  # 生成拒绝响应 用模型自身生成的响应作为拒绝响应
    except Exception as e:
        rejected_resp = "Error: failed to generate response."
        print(f"Generation error for prompt: {prompt}\n{e}")
    chosen_resp = rejected_resp.replace(ORG_NAME, POS_NAME)  # 生成选择响应 这里是通过替换原始响应中的名字来实现的
    chosen = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_resp},
    ]
    rejected = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_resp},
    ]

    return {"chosen": chosen, "rejected": rejected}

# %%
# 将原始数据集转换为DPO的ChatML格式
dpo_ds = raw_ds.map(build_dpo_chatml, remove_columns=raw_ds.column_names)# 应用函数并移除原始列

# %%
dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

# 设置Pandas显示选项以便更好地查看数据
pd.set_option("display.max_colwidth", None)  # 显示每个单元的完整内容
pd.set_option("display.width", 0)  # 自动调整宽度以适应内容

# 显示数据集的前5行
sample_df = dpo_ds.select(range(5)).to_pandas()
print(sample_df)

# %%
if not USE_GPU:
    dpo_ds = dpo_ds.select(range(100))

config = DPOConfig(
    beta=0.2,  # beta参数控制选择和拒绝响应的权重
    per_device_train_batch_size=1,  # 每个设备的训练批次大小
    gradient_accumulation_steps=8,  # 梯度累积步数
    num_train_epochs=1,  # 训练的总轮数
    learning_rate=5e-5,  # 学习率
    logging_steps=2,  # 日志记录步数
)

# %%
# 创建DPO训练器
dpo_trainer = DPOTrainer(
    model=model,  # 模型
    ref_model=None,  # 参考模型（如果有的话）我们通常将其设置为 None，这样它会自动创建原始模型的副本作为参考模型并冻结其权重。
    args=config,  # 训练参数配置
    processing_class=tokenizer,  # 分词器
    train_dataset=dpo_ds,  # 训练数据集
)
# 训练DPO模型
dpo_trainer.train()

# %%
fully_trained_qwen = False
if fully_trained_qwen:
    model, qwen_tokenizer = load_model_and_tokenizer(
        "banghua/Qwen2.5-0.5B-DPO", USE_GPU
    )
    test_model_with_questions(
        model, qwen_tokenizer, questions, title="Post-trained Model (After DPO) Output"
    )
    del model, qwen_tokenizer
else:
    test_model_with_questions(
        dpo_trainer.model,
        tokenizer,
        questions,
        title="Post-trained Model (After DPO) Output",
    )
