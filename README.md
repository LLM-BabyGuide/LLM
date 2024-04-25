# LLM使用

## 计划微调：

Llama2：已完成

ChatGLM

NL2SQL



学习使用将个人数据集和LLM集成。



## 参考资料：

Extended Guide: Instruction-tune Llama 2：https://www.philschmid.de/instruction-tune-llama-2

Stanford Alpaca: An Instruction-following LLaMA Model：https://github.com/tatsu-lab/stanford_alpaca#data-release

Preparing Data for Fine-tunings of Large Language Models：https://heidloff.net/article/fine-tuning-instructions/



## 半精度微调Llama2注意点：

1、LlaMA2模型分词器的padding side要设置为right，否则可能不收敛

```
tokenizer.padding_side='right' # 一定要设置padding_side为right，否则batch大于1时可能不收敛
tokenizer.pad_token = tokenizer.eos_token
```

2、LlaMA2模型的分词器词表的问题，要适当调整数据的最大长度，保证数据内容的完整性：

llama的tokenizer词表很小，中文无法被单个token标识，一个中文字符会被标记为多个token，原文本就扩大了几倍，所以要扩大max_length。

3、LlaMA2模型加载时，需要指定torch dtype为半精度，否则模型将按照fp32进行加载

```
model.half() # 开启半精度
model.enable_input_require_grads()
```

4、prepare_model_for_kbit_training

```
# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
```

5、当启用gradient checkpoint训练时，需要一并调用model.enable input require grads()方法

6、当完全采用fp16半精度进行训练且采用adam优化器时，需要调整优化器中的adam epsilon的值，否则模型无法收敛

```python
args = TrainingArguments(
    output_dir="./my-llama2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=20,
    gradient_checkpointing=True, # 需要model.enable_input_require_grads()
    adam_epsilon=1e-4 # 和半精度舍入有关
)
```

