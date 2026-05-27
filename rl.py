from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score

os.environ['WANDB_MODE'] = 'disabled'

def _to_bool(value):
    """Convert string/bool/int to bool (handles shell script True/False strings)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f"}:
            return False
    return bool(value)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    # value head params (from mcts2)
    value_index_path: str = "",
    value_emb_path: str = "",
    value_head_weight: float = 0.0,
    value_td_gamma: float = 0.99,
    freeze_lm: bool = False,
    ema_value_head_weight: float = 0.0,
    ema_value_head_hidden_mult: int = 2,
    # sibling-grpo params (V-STAR Sec 4.3)
    use_sibling_grpo: bool = False,
    sibling_loss_weight: float = 1.0,
    # VED params (V-STAR Sec 4.2)
    use_ved: bool = False,
    ved_budget_multiplier: float = 1.0,
    ved_init_beam: int = 8,
    ved_lambda: float = 0.1,
    ved_beta_ucb: float = 1.0,
):
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)

    # Convert string bools from shell scripts
    add_gt = _to_bool(add_gt)
    beam_search = _to_bool(beam_search)
    test_during_training = _to_bool(test_during_training)
    dynamic_sampling = _to_bool(dynamic_sampling)
    dapo = _to_bool(dapo)
    gspo = _to_bool(gspo)
    freeze_lm = _to_bool(freeze_lm)
    use_sibling_grpo = _to_bool(use_sibling_grpo)
    use_ved = _to_bool(use_ved)
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=10000)
    train_datasets.append(train_data3)
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    train_data = ConcatDataset(train_datasets)
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}

    prompt2history = {}
    history2target = {}
    
    # Collect prompt2history and history2target from all train datasets
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    # Add eval_data mappings
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]


    def ndcg_rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i%num_generations])
            
            if (i+1)%num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        
        return rewards

    def rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["WANDB_MODE"] = "disabled"

    training_args = GRPOConfig(output_dir=output_dir,
                                save_total_limit=20,
                                eval_strategy="no",
                                max_completion_length=128,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=True,
                                optim="paged_adamw_32bit",
                                lr_scheduler_type="cosine", 
                                save_strategy="epoch",
                                report_to=None,
                                run_name=wandb_run_name,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        # value head params
        value_index_path=value_index_path if value_index_path else None,
        value_emb_path=value_emb_path if value_emb_path else None,
        value_head_weight=value_head_weight,
        value_td_gamma=value_td_gamma,
        freeze_lm=freeze_lm,
        ema_value_head_weight=ema_value_head_weight,
        ema_value_head_hidden_mult=ema_value_head_hidden_mult,
        # sibling-grpo
        use_sibling_grpo=use_sibling_grpo,
        sibling_loss_weight=sibling_loss_weight,
        # VED
        use_ved=use_ved,
        ved_budget_multiplier=ved_budget_multiplier,
        ved_init_beam=ved_init_beam,
        ved_lambda=ved_lambda,
        ved_beta_ucb=ved_beta_ucb,
    )

    trainer.train()

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)
