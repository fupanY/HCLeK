import re
import nltk
import jieba
import torch
import json
import numpy as np
from typing import List, Dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


class PromptCompressor:
    def __init__(
        self,
        model_name,
        device_map: str = "cuda",
        model_config: dict = None,
        word_tokenizer: str = "jieba",
    ):
        """
        Args:
            model_name: str, the name of the model
            device_map: str, the device to load the model, include "cuda", "cpu", "mps". Default is "cuda".
            model_config: dict, the config of the model
            word_tokenizer: str, the word tokenizer, include "jieba", "nltk". Default is "jieba".
        """

        model_config = model_config or {}
        self.model_name = model_name
        self.load_model(model_name, device_map, model_config)

        word_tokenizer_map = {
            "jieba": jieba.lcut,
            "nltk": nltk.word_tokenize,
        }
        self.word_tokenizer = word_tokenizer_map[word_tokenizer]

    def load_model(
        self, model_name: str, device_map: str = "cuda", model_config: dict = None
    ):
        model_config = model_config or {}
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code
        config = AutoConfig.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        if model_config.get("pad_to_left", True):
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = (
                config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            )
        MODEL_CLASS = (
            AutoModelForTokenClassification
            if any("ForTokenClassification" in ar for ar in config.architectures)
            else AutoModelForCausalLM
        )
        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )
        if "cuda" in device_map or "cpu" in device_map:
            model = MODEL_CLASS.from_pretrained(
                model_name,
                torch_dtype=model_config.pop(
                    "torch_dtype", "auto" if device_map == "cuda" else torch.float32
                ),
                device_map="auto",
                config=config,
                ignore_mismatched_sizes=True,
                **model_config,
            )
        else:
            model = MODEL_CLASS.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=model_config.pop("torch_dtype", "auto"),
                pad_token_id=tokenizer.pad_token_id,
                **model_config,
            )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        self.max_position_embeddings = config.max_position_embeddings

    def get_ppl(
        self,
        text: str,
        granularity: str = "sentence",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)

        # ========== model inference ==========
        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values

        # ========== compute loss ==========
        shift_logits = response.logits[..., :-1, :]
        shift_labels = input_ids[..., past_length + 1 : end]
        valid_mask = (attention_mask[:, past_length:end] == 1)[..., :-1]
        valid_mask = valid_mask.flatten()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss = loss[valid_mask]

        # ========== condition mode ==========
        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]
        res = loss.mean() if granularity == "sentence" else loss
        return (res, past_key_values) if return_kv else res

    def get_condition_ppl(
        self,
        text: str,
        question: str,
        condition_in_question: str = "none",
        granularity: str = "sentence",
    ):
        """
        计算三类PPL
        none: p(text)
        before: p(text|question)
        after: p(question|text)
        """
        if condition_in_question == "none":
            return self.get_ppl(text, granularity=granularity)
        elif condition_in_question == "before":
            return self.get_ppl(
                question + text,
                granularity=granularity,
                condition_mode="before",
                condition_pos_id=self.get_token_length(question) - 1,
            )
        elif condition_in_question == "after":
            return self.get_ppl(
                text + question,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(text) - 1,
            )

    def assign_budget(
        self, target_token, token_budget_adjust_ratio, rank_list, budget_type="equal"
    ):
        """
        assign budget to each context based on rank_list.
        Args:
            target_token: float, the target token number
            token_budget_adjust_ratio: float, the ratio to adjust token budget
            rank_list: List[Tuple[int, str]], the rank list
            budget_type: str, the type of budget allocation, include "equal", "square_decay". Default is "equal".
        Returns:
            budget_list: List[int], the budget list
        """

        def _square_decay_allocation(X, A, integer_mode=False):
            """
            平方递减分配算法
            :param X: 文档数量
            :param A: 总token预算
            :param integer_mode: 是否返回整数结果
            :return: 分配结果列表（按重要性从高到低排序）
            """
            # 验证输入
            if X <= 0:
                raise ValueError("文档数量必须大于0")
            if A < 0:
                raise ValueError("预算不能为负数")

            # 计算平方权重总和 (1² + 2² + ... + X²) = X(X+1)(2X+1)/6
            total_weight = X * (X + 1) * (2 * X + 1) // 6
            # 计算浮点分配
            allocations = [A * ((X - i) ** 2) / total_weight for i in range(X)]
            if not integer_mode:
                return allocations

            # 整数分配模式
            # 步骤1：计算基础整数部分
            int_alloc = [int(a) for a in allocations]
            remaining = A - sum(int_alloc)
            # 步骤2：按小数部分从大到小分配剩余token
            decimals = [(allocations[i] - int_alloc[i], i) for i in range(X)]
            decimals.sort(reverse=True)
            # 分配剩余token（每个文档最多+1）
            for i in range(remaining):
                int_alloc[decimals[i][1]] += 1
            return int_alloc

        if budget_type == "equal":
            budget_list = [
                int(target_token * token_budget_adjust_ratio) / len(rank_list)
                for _ in rank_list
            ]
            budget_list[-1] = target_token * token_budget_adjust_ratio - sum(
                budget_list[:-1]
            )
        elif budget_type == "square_decay":
            budget_list = _square_decay_allocation(
                len(rank_list),
                target_token * token_budget_adjust_ratio,
                integer_mode=False,
            )
            new_budget_list = [budget_list[rank] for rank in rank_list]
            budget_list = new_budget_list
        else:
            raise ValueError(f"Invalid budget type: {budget_type}")
        return budget_list

    def sort_texts(
        self,
        texts,
        condition_text,
        condition_ppl_type,
        reverse=False,
        priority_type="ppl",
        mmr_n: int = -1,
        mmr_lambda: float = 0.5,
    ):
        """
        sort texts by priority score
        Args:
            texts: List[str], the texts to be sorted
            condition_text: str, the text to condition on
            condition_ppl_type: str, the type of condition ppl include "before", "after", "none". Default is "after".
            reverse: bool, whether to sort in descending order. Default is False.
            priority_type: str, the type of priority score include "ppl", "mmr". Default is "ppl".
            mmr_n: int, the number of texts to select. Default is -1.
            mmr_lambda: float, the parameter for MMR. Default is 0.5.
        Returns:
            sorted_indices: List[int], the indices of sorted texts
            sorted_texts: List[str], the sorted texts
        """

        def _preprocess(text: str) -> set:
            """text preprocess:
            - remove punctuation
            - tokenize
            """
            text = re.sub(r"[^\w\s]", "", text.lower())
            tokens_set = set(self.word_tokenizer(text))
            return tokens_set

        def _jaccard_similarity(set_a: set, set_b: set) -> float:
            """Jaccard = |intersection| / |union|"""
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            return intersection / union if union > 0 else 0.0

        def _mmr(
            texts,
            perplexities,
            perplexity_ascending: bool = False,
            n: int = 1,
            lambda_param: float = 0.5,
        ) -> List[Dict]:
            """
            MMR: Maximal Marginal Relevance
            MMR = ArgMax_{di} [lambda * p(di,query) - (1-lambda) * max_{dj in selected} sim(di, dj)]

            Args:
                perplexity_ascending: bool, whether to sort perplexities in ascending order. Default is False.
                n: int, the number of documents to select. Default is 1.
                lambda_param: float, the parameter for MMR. Default is 0.5.
            Returns:
                selected_indices: List[int], the selected indices
            """
            if n == -1:
                n = len(texts)
            # 预处理所有文档（缓存词集合）
            processed = [
                (idx, _preprocess(text), perplexity)
                for idx, text, perplexity in zip(range(len(texts)), texts, perplexities)
            ]

            # 按困惑度排序
            processed.sort(key=lambda x: x[2] if perplexity_ascending else -x[2])

            selected = []
            remaining = processed.copy()
            if remaining:
                selected.append(remaining.pop(0))

            while len(selected) < n and len(remaining) > 0:
                max_score = -np.inf
                best_idx = 0

                for idx, remaining_item in enumerate(remaining):
                    text_set = remaining_item[1]
                    perplexity = remaining_item[2]
                    max_jaccard_score = 0
                    for (
                        selected_idx,
                        selected_text_set,
                        selected_perplexity,
                    ) in selected:
                        jaccard_score = _jaccard_similarity(text_set, selected_text_set)
                        if jaccard_score > max_jaccard_score:
                            max_jaccard_score = jaccard_score

                    perplexity_factor = (
                        -1 if perplexity_ascending else 1
                    )  # perplexity越小越好，系数为-1
                    mmr_score = (
                        perplexity_factor * lambda_param * perplexity
                        - (1 - lambda_param) * max_jaccard_score
                    )
                    if mmr_score > max_score:
                        max_score = mmr_score
                        best_idx = idx

                # 将最佳候选移到已选列表
                selected.append(remaining.pop(best_idx))

            return [idx for idx, _, _ in selected]

        # ========== priority score ==========
        if priority_type == "ppl":
            priority_scores = [
                self.get_condition_ppl(
                    text, condition_text, condition_ppl_type, granularity="sentence"
                )
                .cpu()
                .float()
                .numpy()
                .item()
                for text in texts
            ]
            sorted_indices = sorted(
                enumerate(priority_scores),
                key=lambda x: (-1 if reverse else 1) * x[1],
            )
            sorted_indices = [x[0] for x in sorted_indices]
        elif priority_type == "mmr":
            priority_scores = [
                self.get_condition_ppl(
                    text, condition_text, condition_ppl_type, granularity="sentence"
                )
                .cpu()
                .float()
                .numpy()
                .item()
                for text in texts
            ]

            sorted_indices = _mmr(
                texts,
                priority_scores,
                perplexity_ascending=not reverse,
                n=mmr_n,
                lambda_param=mmr_lambda,
            )

        sorted_texts = [texts[i] for i in sorted_indices]
        return sorted_indices, sorted_texts

    def select_texts(self, sorted_texts, texts_length_idx, budget):
        """
        select texts by token budget
        Args:
            sorted_texts: List[Tuple[int, str]], the sorted texts
            texts_length_idx: List[Tuple[int, int]], the indices and lengths of texts
            budget: float, the target token number
        Returns:
            selected_indices: List[int], the indices of selected texts
            remaining_budget: float, the remaining budget
        """
        selected_indices = []
        remaining_budget = budget
        for score_idx, _ in sorted_texts:
            text_idx = texts_length_idx[score_idx][1]
            text_length = texts_length_idx[score_idx][0]

            if remaining_budget - text_length < 0:
                continue

            selected_indices.append(text_idx)
            remaining_budget -= text_length
            if remaining_budget == 0:
                break

        return selected_indices, remaining_budget

    def sort_select_texts(
        self,
        texts,
        budget,
        condition_text,
        condition_ppl_type,
        reverse=True,
        priority_type="ppl",
        mmr_n: int = -1,
        mmr_lambda: float = 0.5,
    ):
        """
        sort sentences by priority score and select sentences by token budget
        Args:
            texts: List[str], the texts to be sorted
            budget: float, the target token number
            condition_text: str, the text to condition on
            condition_ppl_type: str, the type of condition ppl include "before", "after", "none". Default is "after".
            reverse: bool, whether to sort in descending order. Default is True.
            priority_type: str, the type of priority score include "ppl", "mmr". Default is "ppl".
            mmr_n: int, the number of texts to select. Default is -1.
            mmr_lambda: float, the parameter for MMR. Default is 0.5.
        Returns:
            selected_indices: List[int], the indices of selected texts
            selected_texts: List[str], the selected texts
            remaining_budget: float, the remaining budget
        """

        text_length_idx = [
            (self.get_token_length(text), idx) for idx, text in enumerate(texts)
        ]

        sorted_idx_list, sorted_text_list = self.sort_texts(
            texts,
            condition_text,
            condition_ppl_type,
            reverse=reverse,
            priority_type=priority_type,
            mmr_n=mmr_n,
            mmr_lambda=mmr_lambda,
        )
        selected_indices, remaining_budget = self.select_texts(
            [(idx, text) for idx, text in zip(sorted_idx_list, sorted_text_list)],
            text_length_idx,
            budget,
        )

        selected_texts = [texts[i] for i in selected_indices]
        return selected_indices, selected_texts, remaining_budget

    def compress_sentence_level(
        self,
        context: List[str],
        target_token: float,
        token_budget_adjust_ratio: float = 1.4,
        condition_text: str = "",
        condition_ppl_type: str = "after",
        priority_type: str = "ppl",
        reverse: bool = True,
        sort_granularity: str = "context",
        budget_type: str = "equal",
        is_skip_sort: bool = False,
        mmr_n: int = -1,
        mmr_lambda: float = 0.5,
    ):
        """
        compress context by sentence level over all contexts.

        Args:
            context: List[str], the original context
            target_token: float, the target token number
            token_budget_adjust_ratio: float, the ratio to adjust token budget during sentence-level filtering. Default is 1.4.
            condition_text: str, the text to condition on
            condition_ppl_type: str, the type of condition ppl include "before", "after", "none". Default is "after".
            priority_type: str, the type of priority score include "ppl", "mmr". Default is "ppl".
            reverse: bool, whether to sort in descending order. Default is True.
            sort_granularity: str = "context", the granularity of sorting and selecting sentences, include "context", "global". Default is "context". "context" means that each context is sorted and selected separately. "global" means that all sentences are sorted and selected together.
            budget_type: str, the type of budget allocation, include "equal", "square_decay". Default is "equal".
            is_skip_sort: bool, whether to skip sorting. If  True, budget will be allocated according to the original order of the context. Default is False.
            mmr_n: int, the number of texts to select. Default is -1.
            mmr_lambda: float, the parameter for MMR. Default is 0.5.
        """

        def _split_sentences(text):
            # 定义分句的正则表达式模式
            pattern = re.compile(r"[^。！？…]*[。！？…]|[^。！？…]+$")
            sentences = pattern.findall(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

        def _reconstruct_context(sentences_per_context, selected_indices):
            """Rebuild compressed context from selected sentences"""
            compressed_context = []
            global_idx = 0

            for sentences in sentences_per_context:
                # 收集当前context被选中的句子
                selected_sentences = [
                    sent
                    for idx, sent in enumerate(sentences)
                    if (global_idx + idx) in selected_indices
                ]
                compressed_context.append("".join(selected_sentences))

                global_idx += len(sentences)

            return compressed_context
        
        sentences_per_context = [_split_sentences(text) for text in context]
        # 排序和选择
        if sort_granularity == "global":
            # 所有sentence一起排序
            flattened_sentences = [
                s for sentences in sentences_per_context for s in sentences
            ]
            budget = target_token * token_budget_adjust_ratio

            selected_indices, selected_texts, remaining_budget = self.sort_select_texts(
                flattened_sentences,
                budget,
                condition_text,
                condition_ppl_type,
                priority_type=priority_type,
                reverse=reverse,
                mmr_lambda=mmr_lambda,
                mmr_n=mmr_n,
            )
            # 重构
            compressed_context = _reconstruct_context(
                sentences_per_context, selected_indices
            )
        elif sort_granularity == "context":

            # context之间分配budget
            flattened_sentences_per_context = [
                "".join(sentences) for sentences in sentences_per_context
            ]

            if is_skip_sort:
                context_sorted_indices = list(range(len(context)))
            else:
                context_sorted_indices, context_sorted_texts = self.sort_texts(
                    flattened_sentences_per_context,
                    condition_text,
                    condition_ppl_type,
                    reverse=reverse,
                    priority_type=priority_type,
                    mmr_lambda=mmr_lambda,
                    mmr_n=mmr_n,
                )
            budget_list = self.assign_budget(
                target_token,
                token_budget_adjust_ratio,
                context_sorted_indices,
                budget_type=budget_type,
            )

            compressed_context = []
            # sentence-level压缩
            remaining_budget = 0
            for context_idx, budget in zip(context_sorted_indices, budget_list):
                budget = budget + remaining_budget
                sentences = sentences_per_context[context_idx]
                selected_indices, _, remaining_budget = self.sort_select_texts(
                    sentences,
                    budget,
                    condition_text,
                    condition_ppl_type,
                    priority_type=priority_type,
                    reverse=reverse,
                    mmr_lambda=mmr_lambda,
                    mmr_n=mmr_n,
                )
                reconstructed_sentences = _reconstruct_context(
                    [sentences], selected_indices
                )[0]
                compressed_context.append(reconstructed_sentences)

        return compressed_context

    def compress_context_level(
        self,
        context: List[str],
        target_token: float,
        token_budget_adjust_ratio: float = 1.4,
        condition_text: str = "",
        condition_ppl_type: str = "after",
        reverse: bool = True,
    ):
        """
        compress context by context level over all contexts.
        Args:
            context: List[str], the original context
            target_token: float, the target token number
            token_budget_adjust_ratio: float, the ratio to adjust token budget during sentence-level filtering. Default is 1.4.
            condition_text: str, the text to condition on
            condition_ppl_type: str, the type of condition ppl include "before", "after", "none". Default is "after".
            reverse: bool, whether to sort in descending order. Default is True.
        Returns:
            compressed_context: List[str], the compressed context
        """
        budget = target_token * token_budget_adjust_ratio
        selected_indices, selected_context, remaining_budget = self.sort_select_texts(
            context, budget, condition_text, condition_ppl_type, reverse=reverse
        )
        compressed_context = [context[i] for i in selected_indices]
        return compressed_context

    def get_token_length(
        self,
        text: str,
        add_special_tokens: bool = True,
        use_oai_tokenizer: bool = False,
    ):
        """
        add_special_tokens: 分词器会根据模型需求自动添加特殊标记
        """
        if use_oai_tokenizer:
            return len(self.oai_tokenizer.encode(text))
        else:
            return len(
                self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids
            )

    def compress_token_level(
        self,
        context: List[str],
        target_token: float,
        token_budget_adjust_ratio: float = 1.4,
        condition_text: str = "",
        condition_ppl_type: str = "after",
    ):
        """
        compress context by token level separately.
        Args:
            context: List[str], the original context
            target_token: float, the target token number
            token_budget_adjust_ratio: float, the ratio to adjust token budget during sentence-level filtering. Default is 1.4.
            condition_text: str, the text to condition on
            condition_ppl_type: str, the type of condition ppl include "before", "after", "none". Default is "after".
        Returns:
            compressed_context: List[str], the compressed context
        """
        # equal budget
        budget_list = [target_token * token_budget_adjust_ratio / len(context)] * len(
            context
        )
        compressed_context = []
        for text, budget in zip(context, budget_list):
            compressed_context.append(
                self.sort_select_tokens(
                    text, condition_text, condition_ppl_type, budget, reverse=True
                )
            )
        return compressed_context

    def sort_select_tokens(
        self,
        text,
        condition_text,
        condition_ppl_type,
        budget,
        reverse=False,
    ):
        """
        sort texts by priority score
        Args:
            texts: List[str], the texts to be sorted
            condition_text: str, the text to condition on
            condition_ppl_type: str, the type of condition ppl include "before", "after", "none". Default is "after".
            budget: float, the target token number
            reverse: bool, whether to sort in descending order. Default is False.
        Returns:
            compressed_text: str, the compressed text
        """

        priority_scores = (
            self.get_condition_ppl(
                text, condition_text, condition_ppl_type, granularity="token"
            )
            .cpu()
            .float()
            .numpy()
            .item()
        )

        input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        sorted_indices = sorted(
            enumerate(priority_scores),
            key=lambda x: (-1 if reverse else 1) * x[1],
        )
        compressed_input_ids = [input_ids[i] for i, _ in sorted_indices]
        compressed_input_ids = compressed_input_ids[: int(budget)]
        compressed_text = self.tokenizer.decode(compressed_input_ids)
        return compressed_text
