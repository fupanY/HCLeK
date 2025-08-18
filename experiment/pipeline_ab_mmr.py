from HCLeK.prompt_compressor import PromptCompressor
from utils import chat_llm
import json
from tqdm import tqdm
from typing import List
import os
from eval import rouge_score
import numpy as np
import torch
import gc


class LinguaPipeline:
    def __init__(self, compress_model_name):
        self.prompt_compressor = PromptCompressor(model_name=compress_model_name)

    def sort_cases(
        self,
        cases,
        fact,
        articles,
        mmr_n=10,
        mmr_lambda=0.5,
        priority_type="ppl",
        reverse=False,
    ):
        # condition_text = f"\n## 案情事实\n{fact}"
        # condition_text = f"## 相关法条：{articles}\n## 案情事实：{fact}\n"
        # condition_text = f"## 限制\n上述内容有助于法院判决。## 案情事实：{fact}\n## 相关法条\n{articles}"
        condition_text = f"\n## 案情事实：{fact}\n## 相关法条\n{articles}\n## 限制\n能从上述信息给出【案情事实】对应的罪名、刑期、罚金。"
        sorted_idx, sorted_cases = self.prompt_compressor.sort_texts(
            cases,
            condition_text,
            condition_ppl_type="after",
            reverse=reverse,
            mmr_n=mmr_n,
            mmr_lambda=mmr_lambda,
            priority_type="ppl",
        )

        return sorted_cases

    def compress_cases_slevel(
        self,
        cases,
        fact,
        articles,
        target_token,
        sort_granularity="context",
        mmr_n=10,
        mmr_lambda=0.5,
        priority_type="mmr",
        reverse=False,
        budget_type="square_decay",
    ):
        """
        在sentence level上压缩
        """

        # ## 限制\n上述内容涉及罪名、刑期、罚金等量刑信息，是裁判文书的判决部分。## 相关法条\n{articles}
        # condition_text = f"\n## 相关法条\n{articles[0]}\n## 限制\n能从上述信息得到定罪、量刑、罚金的建议。"
        condition_text = f"## 上述内容涉及罪名、刑期、罚金等量刑信息，是裁判文书的判决部分。## 相关法条：{articles[0]}"
        compressed_cases = self.prompt_compressor.compress_sentence_level(
            cases,
            target_token=target_token,
            condition_text=condition_text,
            condition_ppl_type="after",
            sort_granularity=sort_granularity,
            budget_type=budget_type,
            mmr_n=mmr_n,
            mmr_lambda=mmr_lambda,
            is_skip_sort=True,
            priority_type=priority_type,
            reverse=reverse,
            token_budget_adjust_ratio=1.0,
        )

        return compressed_cases

    def pipeline(
        self,
        cases,
        fact,
        articles,
        target_token,
        sort_granularity="context",
        mmr_n=10,
        mmr_lambda=0.8,
        priority_type="mmr",
        reverse=False,
        budget_type="square_decay",
        skip_sort=False,
        skip_compress=False,
    ):
        if skip_sort:
            sorted_cases = cases
        else:
            sorted_cases = self.sort_cases(
                cases,
                fact,
                articles,
                mmr_n=mmr_n,
                mmr_lambda=mmr_lambda,
                priority_type="ppl",
                reverse=reverse,
            )
            sorted_cases = sorted_cases[:10]
        if skip_compress:
            compressed_cases = sorted_cases
        else:
            compressed_cases = self.compress_cases_slevel(
                sorted_cases,
                fact,
                articles,
                target_token,
                sort_granularity,
                mmr_n=mmr_n,
                mmr_lambda=mmr_lambda,
                priority_type=priority_type,
                reverse=reverse,
                budget_type=budget_type,
            )
        return compressed_cases

    def get_target_token(self, raw_text, compress_rate):
        raw_token = self.prompt_compressor.get_token_length(raw_text, False)
        target_token = int(raw_token * compress_rate)
        return target_token


def eval_report(preds, labels, report_file, result_file):
    """
    评估
    """
    rouge_scores = []
    assert len(preds) == len(labels)
    for pred, label in zip(preds, labels):
        if pred == " " or pred == "":
            continue
        rouge_scores.append(rouge_score(pred, label))
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rouge_scores, ensure_ascii=False, indent=2))
    rouge_scores = [item[0] for item in rouge_scores]
    rouge_scores = [
        [
            item["rouge-1"]["r"],
            item["rouge-1"]["p"],
            item["rouge-1"]["f"],
            item["rouge-2"]["r"],
            item["rouge-2"]["p"],
            item["rouge-2"]["f"],
            item["rouge-l"]["r"],
            item["rouge-l"]["p"],
            item["rouge-l"]["f"],
        ]
        for item in rouge_scores
    ]

    rouge_scores = np.array(rouge_scores)
    rouge_scores_mean = rouge_scores.mean(axis=0)
    report_str = f"result: {result_file}\n"
    report_str += f"rouge-1 r: {rouge_scores_mean[0]}, p: {rouge_scores_mean[1]}, f: {rouge_scores_mean[2]}\n"
    report_str += f"rouge-2 r: {rouge_scores_mean[3]}, p: {rouge_scores_mean[4]}, f: {rouge_scores_mean[5]}\n"
    report_str += f"rouge-l r: {rouge_scores_mean[6]}, p: {rouge_scores_mean[7]}, f: {rouge_scores_mean[8]}\n"
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(report_str + "\n")
    print(report_str)
    return rouge_scores


class ExperimentConfig:
    """实验参数配置类"""

    def __init__(self, version, compress_rate=0.5):
        self.version = version
        
        # models
        self.model_name = "/root/autodl-tmp/Qwen2___5-7B-Instruct"
        self.sample_num = 10
        self.generation_model_name = "qwen2.5-7b-instruct-1m"
        
        # data hyperparameters
        self.compress_rate = compress_rate
        self.max_reference_length = 6000
        self.case_num = 10
        self.article_num = 5

        # experiment parameters
        self.mmr_n = -1
        self.mmr_lambda = 0.8
        self.priority_type = "mmr"
        self.reverse = False
        self.budget_type = "square_decay"
        self.skip_sort = False
        self.skip_compress = False
        

        os.makedirs("experiment/res", exist_ok=True)
        # 路径配置
        self.save_paths = {
            "compressed_cases": "experiment/res/{version}_{compress_rate}_compressed_cases.jsonl",
            "response": "experiment/res/{version}_{compress_rate}_response_"
            + self.generation_model_name
            + ".jsonl",
            "result": "experiment/res/{version}_{compress_rate}_result.jsonl",
            "report": "experiment/res/{version}_{compress_rate}_report.txt",
            "record": "experiment/res/{version}_{compress_rate}_record.json",
        }

        # 提示模板
        self.prompt_template = """你是一位资深专业法官。请依据以下的相关案例，精准结合相关法律知识，对给出的案情事实作出合理且公正的判决。请注意，判决结果应包括罪名、量刑、罚金、缓刑（如适用）等信息。如果涉及多个被告人，请分别列出他们的判决结果。请注意，只输出最后的判决结果，无需说明判决理由。\n\n## 相关知识：{articles}\n\n{compressed_reference}\n\n## 案情事实：{fact}\n\n# 判决结果："""

    def get_path(self, key: str) -> str:
        """获取格式化后的存储路径"""
        return self.save_paths[key].format(
            compress_rate=self.compress_rate, version=self.version
        )

    def __repr__(self):
        # 所有参数
        return repr(self.__dict__)


def load_and_prepare_data(config: ExperimentConfig):
    """加载并预处理数据"""
    qa_path = "experiment/data/result.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
    if config.sample_num > 0:
        qa_list = qa_list[: config.sample_num]

    expert_knowledge_path = "experiment/data/expert_knowledge_list.json"
    with open(expert_knowledge_path, "r", encoding="utf-8") as f:
        expert_knowledge = json.load(f)

    return qa_list, expert_knowledge


def get_existing_ids(save_path: str):
    """获取已处理数据的ID列表"""
    if not os.path.exists(save_path):
        return []

    with open(save_path, "r", encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f]


def truncate_texts(texts: List[str], max_length: int) -> List[str]:
    """截断长文本"""
    return [text[-max_length:] if len(text) > max_length else text for text in texts]


def save_result(save_path: str, data: dict):
    """保存处理结果"""
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def run_compression_pipeline(config: ExperimentConfig):
    """运行压缩流程"""
    # 初始化管道
    pipeline = LinguaPipeline(compress_model_name=config.model_name)
    save_path = config.get_path("compressed_cases")

    # 加载数据
    qa_list, expert_knowledge = load_and_prepare_data(config)
    existing_ids = get_existing_ids(save_path)

    # 处理每个案例
    for qa in tqdm(qa_list):
        if qa["id"] in existing_ids:
            continue

        # 准备参考材料
        fact = qa["fact"]
        articles = [a["article"] for a in qa["recall_articles"]][: config.article_num]
        cases = [c["case"]["qw"] for c in qa["recall_cases"]][: config.case_num]

        reference = cases + expert_knowledge
        reference = truncate_texts(reference, config.max_reference_length)

        # 执行压缩
        combined_text = "\n".join(reference)
        target_token = pipeline.get_target_token(combined_text, config.compress_rate)
        compressed = pipeline.pipeline(
            reference,
            fact,
            articles,
            target_token,
            sort_granularity="context",
            mmr_n=config.mmr_n,
            mmr_lambda=config.mmr_lambda,
            priority_type=config.priority_type,
            reverse=config.reverse,
            budget_type=config.budget_type,
        )

        # 保存结果
        save_result(save_path, {"id": qa["id"], "compressed_reference": compressed})
    del pipeline
    gc.collect()


def run_generation_pipeline(config: ExperimentConfig):
    """运行答案生成流程"""

    compressed_path = config.get_path("compressed_cases")
    save_path = config.get_path("response")

    # 加载数据
    qa_list, _ = load_and_prepare_data(config)
    existing_ids = get_existing_ids(save_path)

    # 加载压缩结果
    if os.path.exists(compressed_path):
        with open(compressed_path, "r", encoding="utf-8") as f:
            compressed_refs = [json.loads(line)["compressed_reference"] for line in f]
    else:
        compressed_refs = []

    # 生成回答
    for i, qa in enumerate(tqdm(qa_list)):
        if qa["id"] in existing_ids:
            continue

        # 构建提示
        articles = "\n".join(
            [a["article"] for a in qa["recall_articles"]][: config.article_num]
        )
        tmp_compressed_ref = compressed_refs[i]
        # 删去空串
        tmp_compressed_ref = [ref for ref in tmp_compressed_ref if ref]
        compressed_ref = "\n".join(tmp_compressed_ref)
        prompt = config.prompt_template.format(
            fact=qa["fact"], articles=articles, compressed_reference=compressed_ref
        )

        # 生成并保存结果
        response = chat_llm(messages=prompt, model_name=config.generation_model_name)
        response = response.replace("\n", "")
        with open("experiment/res/record.txt", "a", encoding="utf-8") as f:
            f.write(repr(prompt) + "\n")
            f.write(repr(str(response)) + "\n")
        # response = response[0].outputs[0].text
        save_result(
            save_path, {"id": qa["id"], "response": response, "gt": qa["result"]}
        )

    gc.collect()
    # eval
    # 根据id取出preds
    labels = []
    preds = []
    with open(save_path, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]

    for pred in preds:
        for qa in qa_list:
            if qa["id"] == pred["id"]:
                labels.append(qa["result"])
                break
    preds = [pred["response"] for pred in preds]
    eval_report(preds, labels, config.get_path("report"), config.get_path("result"))


if __name__ == "__main__":
    version = "main"
    config = ExperimentConfig(version=version)
    with open(config.get_path("record"), "w", encoding="utf-8") as f:
        f.write(repr(config))

    for compress_rate in [0.05, 0.1, 0.2, 0.5]:
        torch.cuda.empty_cache()
        gc.collect()
        config = ExperimentConfig(
            compress_rate=compress_rate,
            version=version,
        )
        config.mmr_lambda = 1.0
        config.sample_num = 531
        run_compression_pipeline(config)
        run_generation_pipeline(config)
