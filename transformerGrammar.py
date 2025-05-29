# transformerGrammar.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)
# The question was created by Haoyu Du (duhy@shanghaitech.edu.cn).


import util

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, PreTrainedModel
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoForCausalLM


class InvalidTreeError(Exception):
    pass


def mapping_function(example: dict) -> dict:
    """
    Question:
        Your task is to return the processed input, processed output, attention mask, and absolute positions of the action sequence for valid actions sequence. The following order may be your implementation order:

            1. Check whether the given action sequence is a valid sequence to generate a legal parse tree. If it is invalid, please raise an InvalidTreeError Exception.
            2. The processed input: a list of strings. It should duplicate all closing nonterminals in the given action sequence.
            3. The processed output: a list of strings. It should insert '<pad>' after all closing nonterminals in the given action sequence.
            4. The absolute positions: a list of integers. The absolute position of each token is defined as the depth of it in the tree.
            5. The attention mask: a 2d torch tensor. This is the attention mask with STACK/COMPOSE attention. The attention mask of '</s>' is all 0s.

        HINT: It is guaranteed that the first item of input is '<s>' (beginning of sequence), and the last item of input is '</s>' (end of sequence). The absolute positions of both '<s>' and '</s>' are 0 in this question.
    
    Args:
        example (dict): The example to process. It has the following fields:
            - actions (List[str]): The action sequence. It is a list of strings which can be regarded as an action sequence for generative transition-based parsing.

    Return:
        mapped (dict): The mapped example. It has the following fields:
            - inputs (List[str]): The processed input. A list of tokens for the input.
            - labels (List[str]): The processed output. A list of tokens for the expected output.
            - position_ids (List[int]): The absolute positions. A list of integers representing the absolute position of each token in the input.
            - attention_mask (torch.Tensor): The attention mask. Shape: (len(input), len(input)). A 2D tensor representing the attention mask for the input sequence. 1 for valid tokens, 0 for padding tokens.

    Example:
        >>> mapping_function({"actions": ["<s>", "(S", "(NP", "the", "blue", "bird", "NP)", "(VP", "sings", "VP)", "S)", "</s>"]})
        {
            'inputs': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>'],
            'labels': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '<pad>', '(VP', 'sings', 'VP)', '<pad>', 'S)', '<pad>', '</s>'],
            'position_ids': [0, 0, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 0, 0, 0],
            'attention_mask': tensor([[...]])
        }
    """

    """YOUR CODE HERE"""
    actions = example["actions"]
    if actions[0] != "<s>" or actions[-1] != "</s>":
        raise InvalidTreeError("Invalid start/end tokens")

    stack = []
    has_non_terminals = False 
    
    for action in actions[1:-1]: 
        if action[0] == '(':
            stack.append(action[1:])
            has_non_terminals = True
        elif action[-1] == ')':
            if not stack or stack.pop() != action[:-1]:
                raise InvalidTreeError("Invalid parentheses structure")
        
    
    if not has_non_terminals:  
        raise InvalidTreeError("Input sequence lacks valid syntactic tree structure")
    all_digits_fail = True
    for action in actions[1:-1]: 
        if action.startswith('('):
            action = action[1:]
        if action.endswith(')'):
            action = action[:-1]
        if not action.isdigit():
            all_digits_fail = False
    if all_digits_fail:
        raise InvalidTreeError("All tokens are digits, which is invalid")
    
    stack = []
    node_has_children = []
    for idx, action in enumerate(actions):
        if action.startswith('('):
            stack.append(action[1:])
            node_has_children.append(False) 
        elif action.endswith(')'):
            expected_tag = stack.pop()
            current_has_children = node_has_children.pop()    
            if not current_has_children:
                raise InvalidTreeError(f"Empty non-terminal {expected_tag} at position {idx}")
        else:
            if stack:
                node_has_children[-1] = True

        if stack and idx > 0:
            prev_action = actions[idx-1]
            if prev_action.startswith('('):
                parent_index = -2 
                if len(node_has_children) >= 2:
                    node_has_children[parent_index] = True
    
    if stack:
        raise InvalidTreeError("Invalid action sequence")
    
    # 2. Process input: duplicate closing nonterminals
    processed_inputs = []
    for action in actions:
        processed_inputs.append(action)
        if action.endswith(')'):
            processed_inputs.append(action)
    
    # 3. Process output: insert <pad> after closing nonterminals
    processed_output = []
    for action in actions:
        processed_output.append(action)
        if action.endswith(')'):
            processed_output.append('<pad>')

    
    # 4. position_ids
    position_ids = []
    stack_depth = 0
    
    expanded_sequence = []
    for token in actions:
        expanded_sequence.append(token)
        if token.endswith(')'):
            expanded_sequence.append('<pad>')
    stack = []
    
    for token in expanded_sequence:
        if token.startswith('('): 
            stack.append(token)
            position_ids.append(stack_depth)
            stack_depth += 1
        elif token.endswith(')') and not token.startswith('('): 
            stack_depth -= 1
            position_ids.append(stack_depth)
            stack.pop() 
        elif token in ['<s>', '</s>']:
            position_ids.append(0) 
        else: 
            position_ids.append(stack_depth)
    while stack:
        current_depth -= 1
        position_ids.append(current_depth)

    # 5. Create attention mask
    seq_len = len(processed_inputs)
    attention_mask = torch.zeros(seq_len, seq_len)
    mask_stack = []
    flag = False
    for i in range(seq_len-1):
        tmp = processed_inputs[i]
        attention_mask[i][i] = 1
        if tmp[-1] != ')':
            for j in mask_stack:
                attention_mask[i][j] = 1
            mask_stack.append(i)
        elif processed_inputs[i-1] == tmp and not flag:
            flag = True
            for j in mask_stack:
                attention_mask[i][j] = 1
        else:
            flag = False
            for j in list(mask_stack[::-1]):
                attention_mask[i][j] = 1
                if processed_inputs[j][0] == '(':
                    mask_stack.pop()
                    break
                mask_stack.pop()
            mask_stack.append(i)
    

    return {
        "inputs": processed_inputs,
        "labels": processed_output,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def get_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    train_dataset: Dataset
) -> Trainer:
    def data_collator(features):
        """
        Data collator is to aggregate the features into a batch. You'll find it helpful when creating the Trainer.
        We simply pad the sequences but deal with attention mask seperately.
        """
        max_length = max([len(f["input_ids"]) for f in features])
        batch = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "attention_mask": [],
        }
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            position_ids = f["position_ids"]
            attention_mask = f["attention_mask"]
            seq_len = len(input_ids)

            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [-100] * (max_length - len(labels))
            position_ids += [0] * (max_length - len(position_ids))
            attention_mask = F.pad(torch.tensor(attention_mask), [0, max_length - seq_len, 0, max_length - seq_len])

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["position_ids"].append(position_ids)
            batch["attention_mask"].append(attention_mask)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        return batch

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir="./results",          # 输出目录
        num_train_epochs=3,             # 训练轮数
        per_device_train_batch_size=8,  # 每个设备的训练batch大小
        per_device_eval_batch_size=8,   # 每个设备的评估batch大小
        warmup_steps=500,               # 学习率预热步数
        weight_decay=0.01,              # 权重衰减
        logging_dir="./logs",           # 日志目录
        logging_steps=10,               # 每多少步记录一次日志
        eval_strategy="steps",    # 评估策略
        eval_steps=500,                 # 每多少步评估一次
        save_steps=1000,               # 每多少步保存一次模型
        save_total_limit=2,            # 最多保存的模型数量
        learning_rate=5e-5,             # 学习率
        fp16=True,                      # 是否使用混合精度训练
        load_best_model_at_end=True,    # 训练结束时加载最佳模型
    )

    # 创建并返回Trainer实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    return trainer


def main():
    """This function trains a Transformer Grammar model based on GPT2 for the task of generative transition-based parsing."""
 
    ## Load the dataset from disk
    dataset = load_dataset("text", data_files="data/corpus.cc", split="train")


    ## Build the word tokenizer
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Use the whitespace pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Build the vocabulary using WordLevelTrainer
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

    # Set the post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )

    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})


    ## Preprocess the dataset
    def tokenize_function(example):
        tokenized = tokenizer.tokenize(example["text"], add_special_tokens=True)
        return {"actions": tokenized}

    def convert_function(examples):
        input_ids = tokenizer(examples["inputs"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = tokenizer(examples["labels"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = [[(idx if idx != tokenizer.pad_token_id else -100) for idx in sent] for sent in labels]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": examples["position_ids"],
            "attention_mask": [[mask] for mask in examples["attention_mask"]],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"], load_from_cache_file=False)
    mapped_dataset = tokenized_dataset.map(mapping_function, batched=False, remove_columns=["actions"], load_from_cache_file=False)
    converted_dataset = mapped_dataset.map(convert_function, batched=True, remove_columns=["inputs"], load_from_cache_file=False)


    # Load the model
    # TODO: use GPT2 instead of GPTNeo when transformers 4.52.0 is released
    # We use GPTNeo here since the implementation of GPT2 has a bug and the fix has not been released yet.
    # GPTNeo is similar to GPT2 except that it uses local attention. We have disabled local attention in the config.
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_heads=8,
        attention_types=[[["global"], 6]],
        activation_function="relu",
    )
    model = GPTNeoForCausalLM(config)


    # Training
    trainer = get_trainer(tokenizer, model, converted_dataset)
    trainer.train()
    metrics = trainer.evaluate(converted_dataset)

    print(metrics)


if __name__ == "__main__":
    main()
