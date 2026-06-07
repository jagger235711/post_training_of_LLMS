# %% 导入所有库
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# %% 一些辅助函数


def generate_responses(
    model, tokenizer, user_message, system_message=None, max_new_tokens=100
):
    """

    接受用户输入，调用模型生成响应


    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        user_message (_type_): _description_
        system_message (_type_, optional): _description_. Defaults to None.
        max_new_tokens (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(  # 转换为模板格式
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # qwen3特殊设置
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # temperature=0.7,
            # top_p=0.9,
        )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]  # 从输出中截取响应
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return response


# %%
def test_model_with_questions(
    model, tokenizer, questions, system_message=None, title="Model Output"
):
    """
    使用批量输入测试模型

    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        questions (_type_): _description_
        system_message (_type_, optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to "Model Output".
    """

    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")


# %%
# 加载模型并定义 tokenizer
def load_model_and_tokenizer(model_name, use_gpu=False):

    # 加载基座模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_gpu:
        model.to("cuda")

    # 定义默认的 chat tempalte
    # 如你所见，在使用 LLM 时，模型看到的信息是由 chat message list 转换而来的一个 token sequence，从而继续一个一个生成下一个 token。

    # - 主要是学习输入数据的格式和知识，一般不用于调整风格
    # - 通过在模板中添加prompt可以调整风格，会导致问答能力下降。容易答非所问
    tokenizer.chat_template = """
        {% for message in messages %}
        {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
        {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
        {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} Meow~ <|endoftext|>
        {% endif %}
        {% endfor %}

    """

    # 将用于填充的 pad token 设置为用于结尾的 eos token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_chat_example(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


# %%
# 可视化数据集
def display_dataset(dataset):
    rows = []
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        user_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "user"
        )
        assistant_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "assistant"
        )
        rows.append({"User Prompt": user_msg, "Assistant Response": assistant_msg})

    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", None)  # Avoid truncating long strings
    # display(df)
    print(df.to_string())


# %% 加载 Qwen3-0.6B 的 Base 模型并针对简单问题进行测试
USE_GPU = torch.cuda.is_available()

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?",
    "爸爸的弟弟我应该如何称呼？",
]

# %%
model, tokenizer = load_model_and_tokenizer(
    "Qwen/Qwen3-0.6B-Base", USE_GPU
)  # 用的base模型，没有微调过的

test_model_with_questions(  # 测试模型输出结果
    model, tokenizer, questions, title="Base Model (Before SFT) Output"
)

del (
    model,
    tokenizer,
)  # del 是 Python 中的删除（释放）对象引用的关键字，它本身不会直接销毁对象，但可以用来切断变量名和内存中对象的绑定关系，让对象满足被垃圾回收（GC）的条件。

# %%
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B", USE_GPU)  # 微调过的

test_model_with_questions(
    model, tokenizer, questions, title="Base Model (After SFT) Output"
)

del model, tokenizer

# %%在小模型上进行SFT
model_name = "HuggingFaceTB/SmolLM2-135M"
model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)

# %%
train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
if not USE_GPU:
    train_dataset = train_dataset.select(range(100))

display_dataset(train_dataset)

# %%SFT训练器配置

# SFTTrainer 设置
sft_config = SFTConfig(
    output_dir="./sft_outputs",
    learning_rate=8e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,  # 每块 GPU 的 batch size。
    gradient_accumulation_steps=8,  # 梯度累积次数。执行梯度下降前的累积步数
    gradient_checkpointing=False,  # 启用梯度检查点机制，以降低训练期间的内存使用量，但会以训练速度变慢为代价。
    logging_steps=2,  # 每两个 step 打印一次 log。
    # max_length=1024,
    packing=True,  # 是否启用 packing。packing 是一种技术，可以将多个训练样本组合成一个更长的输入序列，以更有效地利用 GPU 内存和提高训练效率。
    use_cpu=not USE_GPU,
)

# %%
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # 训练前使用的数据预处理类
    formatting_func=lambda example: format_chat_example(example, tokenizer),
)
sft_trainer.train()

# %%
if not USE_GPU:
    sft_trainer.model.to("cpu")
test_model_with_questions(
    sft_trainer.model, tokenizer, questions, title="Base Model (After SFT) Output"
)

# %%
print(tokenizer.chat_template)
print(USE_GPU)
