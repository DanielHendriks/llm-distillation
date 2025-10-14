import logging
from datasets import load_dataset, load_from_disk, concatenate_datasets
from .generation_utils import RationaleGenerationStringContainer
from .generation_utils import CritiqueStringContainer
from datasets import disable_caching
disable_caching()


class StrategyQADatasetLoader:
    """
    StrategyQA Dataset Loader for Strategic Question Answering with explanations.
    
    Similar to ESNLIDatasetLoader but adapted for strategic reasoning tasks
    with question/answer pairs and boolean responses. Features hybrid loading that tries Arrow format
    first and falls back to HuggingFace if not available.
    """
    
    def __init__(self, sample=1) -> None:
        # Dataset specific properties
        self.dataset_name = 'strategyqa'
        self.source_dataset_name = "ChilleD/StrategyQA"
        self.dataset_version = None  # Uses default version
        self.has_valid = False  # StrategyQA has train and test splits (no validation)
        
        # Answer mapping for boolean responses
        self.answer_map = {True: 'yes', False: 'no'}
        
        # Settings for generation processes
        self.sample = sample
        
        # Helper and assertion variables
        self.ready_for_rationalizing = False
        self.ready_for_critique = False
        self.ready_for_revision = False
        self.ready_for_cfrationalizing = False
        self._seed = 42
    
    ### Internal methods for hybrid loading
    def _load_from_arrow(self, path):
        """Load dataset from Arrow format (with error handling)"""
        try:
            # Try loading as single dataset first
            return load_from_disk(path)
        except:
            # Fall back to sharded format like CQA
            num_shards = 10
            return concatenate_datasets([
                load_from_disk(f"{path}/shard_{shard_idx}")
                for shard_idx in range(num_shards)
            ])
    
    def dep_load_from_huggingface(self, split='train'):
        """Load StrategyQA dataset from HuggingFace as fallback"""
        ds = load_dataset(self.source_dataset_name)
        # Note: split parameter available for future use
        return ds

    def _load_from_huggingface(self, split='train'):
        """Load StrategyQA dataset from HuggingFace as fallback"""
        ds = load_dataset(self.source_dataset_name)
        
        # StrategyQA has train and test splits, sample if needed
        if self.sample < 1 and split in ds:
            ds[split] = ds[split].shuffle(seed=42).select(range(int(len(ds[split]) * self.sample)))
        
        return ds
    
    def _infer_split_from_path(self, path):
        """Infer which split to load from HuggingFace based on path"""
        path_lower = str(path).lower()
        if 'test' in path_lower:
            return 'test'
        else:
            return 'train'
    
    def _get_safe_rationale(self, example, target_rationale):
        """Safely get rationale with fallback logic"""
        # Try target rationale first
        if target_rationale in example and example[target_rationale]:
            return example[target_rationale]
    
    ### Main training preparation method (ESNLI-compatible interface)
    def prepare_for_training(self, path, model_type, tokenizer, max_length, target_rationale, do_tokenize=True, truncation=True, padding=True):
        """
        Prepare StrategyQA dataset for training with hybrid loading.
        
        Args:
            path: Path to preprocessed Arrow format data
            model_type: Training mode - currently supports 'task_prefix'
            tokenizer: HuggingFace tokenizer for encoding
            max_length: Maximum sequence length
            target_rationale: Which rationale to use ('facts', 'llama_rationale', etc.)
            do_tokenize: Whether to tokenize the data
        
        Returns:
            Tokenized dataset ready for training
        """
        
        # Hybrid loading: try Arrow format first, fall back to HuggingFace
        dataset = self._load_from_arrow(path)
        print(f"Loaded from Arrow format: {path}")
        
        # Apply sampling if specified
        if self.sample < 1:
            dataset = dataset.train_test_split(train_size=self.sample)['train']
            print(f"Applied sampling: {int(self.sample*100)}% of data")

        # Filter out examples without valid answers
        dataset = dataset.filter(lambda x: 'answer' in x and x['answer'] is not None)
        
        # Apply appropriate template based on model type
        if model_type == "task_prefix":
            logging.debug("Entered task prefix mode...")
            # Add this before processing to understand your data better:
            answer_counts = {True: 0, False: 0}
            for example in dataset:
                answer = example['answer']
                if answer in answer_counts:
                    answer_counts[answer] += 1
            
            print(f"Answer distribution: {answer_counts}")
            
            dataset = dataset.map(
                lambda x: self._template_for_taskprefix_training(
                    example=x, target_rationale=target_rationale
                ),
                desc="Applying task prefix templates"
            )
            
            if do_tokenize:
                # Tokenize dataset
                tokenized_datasets = dataset.map(
                    lambda x: self._tokenize_function_taskprefix(
                        examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding
                    ),
                    batched=True,
                    desc="Tokenizing dataset"
                )
                # Select relevant columns for training
                tokenized_datasets = tokenized_datasets.select_columns([
                    'multitask_predict_input_encoded_input_ids', 
                    'multitask_predict_input_encoded_attention_mask', 
                    'multitask_predict_label_encoded_input_ids', 
                    'multitask_predict_label_encoded_attention_mask', 
                    'multitask_explain_input_encoded_input_ids', 
                    'multitask_explain_input_encoded_attention_mask', 
                    'multitask_explain_label_encoded_input_ids', 
                    'multitask_explain_label_encoded_attention_mask'
                ])
            else:
                tokenized_datasets = dataset
                
        elif model_type == "standard":
            logging.debug("Entered standard mode...")
            # Apply template
            dataset = dataset.map(lambda x: self._template_for_stndrd_training(x, target_rationale=target_rationale))
            
            if do_tokenize:
                # Tokenize dataset
                tokenized_datasets = dataset.map(
                    lambda x: self._tokenize_function_standard(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                    batched=True,
                    desc="Tokenizing dataset"
                )
                # Select relevant columns
                tokenized_datasets = tokenized_datasets.select_columns(
                    ['input_ids','labels','attention_mask']
                )
            else:
                tokenized_datasets = dataset
                
        elif model_type == "counterfactual_prefix":
            logging.debug("Entered counterfactual prefix mode...")
            # Apply template
            dataset = dataset.map(lambda x: self._template_for_counterfactual_training(x, target_rationale=target_rationale))
            
            if do_tokenize:
                # Tokenize dataset
                tokenized_datasets = dataset.map(
                    lambda x: self._tokenize_function_counterfactual(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                    batched=True,
                    desc="Tokenizing dataset"
                )
                # Select relevant columns - TODO: implement column selection for counterfactual mode
                # tokenized_datasets = tokenized_datasets.select_columns([...])
            else:
                tokenized_datasets = dataset
                
        elif model_type == "both":
            logging.debug("Entered both (multitask + counterfactual) mode...")
            # Apply both templates
            dataset = dataset.map(lambda x: self._template_for_counterfactual_training(x, target_rationale=target_rationale))
            dataset = dataset.map(lambda x: self._template_for_taskprefix_training(example=x, target_rationale=target_rationale))
            
            if do_tokenize:
                # Tokenize with both functions
                tokenized_datasets = dataset.map(
                    lambda x: self._tokenize_function_counterfactual(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                    batched=True,
                    desc="Tokenizing counterfactual"
                )
                tokenized_datasets = tokenized_datasets.map(
                    lambda x: self._tokenize_function_taskprefix(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                    batched=True,
                    desc="Tokenizing task prefix"
                )
                # Select relevant columns - TODO: implement column selection for both mode
                # tokenized_datasets = tokenized_datasets.select_columns([...])
            else:
                tokenized_datasets = dataset
                
        else:
            raise NotImplementedError(f"Model type '{model_type}' not yet implemented for StrategyQA")
        
        return tokenized_datasets
    
    ### Template methods for task prefix training
    def _template_for_taskprefix_training(self, example, target_rationale='facts'):
        """
        Template StrategyQA data for task_prefix training mode.
        
        Creates multitask format with [predict] and [explain] prefixes compatible with
        existing training pipeline. Supports flexible rationale selection.
        """
        # Create question format for strategic reasoning
        question = example['question']
        
        # Get the answer as yes/no
        answer_text = self.answer_map[example['answer']]
        
        # Prediction task: model should output "The answer is [yes/no]"
        example['multitask_predict_input'] = f"[predict] Question: {question}"
        example['multitask_predict_label'] = f"The answer is {answer_text}"
        
        # Explanation task: model should output the rationale
        example['multitask_explain_input'] = f"[explain] Question: {question}"
        
        # Use flexible rationale selection with fallback
        example['multitask_explain_label'] = self._get_safe_rationale(example, target_rationale).split('\n')[0].strip()
        
        return example
    
    def _tokenize_function_taskprefix(self, examples, tokenizer, max_length, truncation=True, padding=True):
        """
        Tokenize for task_prefix training mode.
        
        Creates the exact column structure expected by MultitaskDataCollator.
        """
        model_inputs = {
            "multitask_predict_input_encoded": tokenizer(
                examples["multitask_predict_input"], 
                max_length=max_length, 
                truncation=truncation, 
                padding=padding
            ),
            "multitask_predict_label_encoded": tokenizer(
                examples["multitask_predict_label"], 
                max_length=300, 
                truncation=truncation, 
                padding=padding
            ),
            "multitask_explain_input_encoded": tokenizer(
                examples["multitask_explain_input"], 
                max_length=max_length, 
                truncation=truncation, 
                padding=padding
            ),
            "multitask_explain_label_encoded": tokenizer(
                examples["multitask_explain_label"], 
                max_length=300, 
                truncation=truncation, 
                padding=padding
            ),
        }
        
        # Flatten the tokenized dictionaries to match expected column names
        model_inputs_flattened = {}
        for key_outer in model_inputs:
            for key_inner in model_inputs[key_outer]:
                model_inputs_flattened[f"{key_outer}_{key_inner}"] = model_inputs[key_outer][key_inner]

        return model_inputs_flattened

    ### Template methods for standard training
    def _template_for_stndrd_training(self, example, target_rationale=None):
        """
        Template StrategyQA data for standard training mode.
        
        Creates simple question/answer format compatible with standard fine-tuning.
        """
        # Create question format
        question = example['question']
        answer_text = self.answer_map[example['answer']]
        
        example['input'] = f"[question] {question}"
        
        # Use flexible rationale selection with fallback
        rationale = self._get_safe_rationale(example, target_rationale) if target_rationale else ''
        example['label'] = f"[answer] The answer is {answer_text}. {rationale}"
        
        return example
    
    def _tokenize_function_standard(self, examples, tokenizer, max_length, truncation=True, padding=True):
        """
        Tokenize for standard training mode.
        
        Creates standard input_ids/labels structure for basic fine-tuning.
        """
        model_inputs = tokenizer(
            examples['input'],
            max_length=max_length,
            truncation=truncation
        )

        label_output_encodings = tokenizer(text_target=examples['label'], max_length=256, truncation=truncation, padding=padding)
        model_inputs['labels'] = label_output_encodings['input_ids']

        return model_inputs
    
    ### Template methods for counterfactual training
    def _template_for_counterfactual_training(self, example, target_rationale=None):
        if target_rationale is not None:
            _correct_rationale_name = target_rationale
        else:
            _correct_rationale_name = "facts"
        
        question = example["question"]
        
        # Normalize gold answer
        gold = example["answer"]
        if isinstance(gold, str):
            gold = gold.lower() == "yes"
        correct_answer_text = "yes" if gold else "no"
        incorrect_answer_text = "no" if gold else "yes"
        
        # Factual
        example["correct_answer_input"] = f"[factual] Question: {question}"
        correct_rationale = self._get_safe_rationale(example, _correct_rationale_name).split('\n')[0].strip()
        example["correct_answer_label"] = f"[factual] Answer: {correct_rationale} So the answer is {correct_answer_text}."
        
        # Counterfactual
        cf_rationale_col = (
            "counterfactual_prompt_answer_1_rationale" if gold else "counterfactual_prompt_answer_2_rationale"
        )
        if cf_rationale_col in example and example[cf_rationale_col]:
            rational = example[cf_rationale_col].split('\n')[0].strip()
            example["false_answer1_input"] = f"[counterfactual] Question: {question}\nRationale: {rational}"
            example["false_answer1_label"] = f"[counterfactual] So the answer is {incorrect_answer_text}."
        
        return example

    
    def _tokenize_function_counterfactual(self, examples, tokenizer, max_length, truncation=True, padding=True):
        """
        Tokenize for counterfactual training mode.
        
        Creates complex tokenization structure for factual/counterfactual pairs.
        """
        # Base tokenization for correct answers
        model_inputs = {
            'correct_answer_input_encoded': tokenizer(examples['correct_answer_input'], max_length=max_length, truncation=truncation, padding=padding),
            'correct_answer_label_encoded': tokenizer(examples['correct_answer_label'], max_length=256, truncation=truncation, padding=padding),
        }
        
        # Add false answer if it exists
        if 'false_answer1_input' in examples and examples['false_answer1_input']:
            model_inputs['false_answer1_input_encoded'] = tokenizer(examples['false_answer1_input'], max_length=max_length, truncation=truncation, padding=padding)
            model_inputs['false_answer1_label_encoded'] = tokenizer(examples['false_answer1_label'], max_length=256, truncation=truncation, padding=padding)
        
        # Flatten the tokenized dictionaries
        model_inputs_flattened = {}
        for key_outer in model_inputs:
            for key_inner in model_inputs[key_outer]:
                model_inputs_flattened[f"{key_outer}_{key_inner}"] = model_inputs[key_outer][key_inner]

        return model_inputs_flattened


    ### methods for rationalizing
    def prepare_to_rationalize(self, prompting_mode: list, direction: list):
        self.dataset = self._load_from_huggingface()
        self.dataset = self.dataset.filter(lambda x: 'answer' in x and x['answer'] is not None)
    
        if not self.ready_for_rationalizing:
            string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
            few_shot_prefix = string_container.get_few_shot_example()
    
            def add_prompts(batch):
                out = {}
                for mode in prompting_mode:
                    for dire in direction:
                        texts = []
                        for question, answer in zip(batch["question"], batch["answer"]):
                            answer_text = self.answer_map[answer]
                            text = ""
                            if mode == 'few_shot':
                                text += few_shot_prefix + "\n"
                            text += f"{question}\nThe answer is {answer_text}."
                            text += "\nBriefly explain the reasoning behind this answer."
                            texts.append(text)
                        out[f"{mode}_{dire}_prompt"] = texts
                return out
    
            self.dataset = self.dataset.map(add_prompts, batched=True, batch_size=1000)
            self.ready_for_rationalizing = True

            
    def _load_from_json(self):
        data_files = {f'{split}': f'../datasets/strategyqa/{split}.json' for split in ['train', 'test']}
        return load_dataset('json', data_files=data_files)
            
    def _template_for_rationale(self, example, prompting_mode, direction):
        # Create question format for strategic reasoning
        question = example['question']
        answer_text = self.answer_map[example['answer']]
        
        # get random affixes for direction
        string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
        # affixes = string_container.get_random_affixes(direction=direction)

        text = ""
        if prompting_mode == 'few_shot':
            text += string_container.get_few_shot_example()
            text += "\n"
        text += f"{question}\nThe answer is {answer_text}."
        text += "\nBriefly explain the reasoning behind this answer."
        
        example[f'{prompting_mode}_{direction}_prompt'] = text
        return example 
    
    
    ### methods for critiquing
    def prepare_for_critique(self, input_name, n_examples_critique, **kwargs):
        self.dataset = self._load_rationalized_dataset(input_name=input_name)
        
        if self.ready_for_critique == False:
            self.dataset = self.dataset.map(lambda x: self._template_for_critique(example=x, n_examples_critique=n_examples_critique))
            self.ready_for_critique = True
            logging.info("Dataset is ready for critique.")

    def _load_rationalized_dataset(self, input_name, _n_shards = 10):
        try: 
            dataset = load_from_disk(f"../llm_outputs/rationales/{input_name}")
        except FileNotFoundError:
            dataset = concatenate_datasets([load_from_disk(f"../llm_outputs/rationales/{input_name}/shard_{shard_idx}") for shard_idx in range(_n_shards)])
        if self.sample < 1:
            logging.info("Loaded {sample} % of the data with rationales.".format(sample=int(self.sample*100)))
            return dataset.train_test_split(train_size=self.sample)['train']    
        else:
            logging.info("Loaded 100 % of the data with rationales.")
            return dataset
            
    def _template_for_critique(self, example, n_examples_critique):
        assert n_examples_critique > 0
        assert n_examples_critique < 8
        csc = CritiqueStringContainer(dataset_name=self.dataset_name)
        
        few_shot_examples = "\n\n".join(csc.get_random_critique_examples(n_examples_critique))
        critique_request, revision_request = csc.get_random_critique_revision_request_pair()
        
        critique_string = f"{few_shot_examples}\n\nQuestion: {example['question']}\nAnswer: The answer is {self.answer_map[example['answer']]}. {example['few_shot_positive_rationale']}\n{critique_request}\nCritique: "
        example['few_shot_critique_prompt'] = critique_string
        example['few_shot_revision_instruction'] = revision_request
        
        return example
    
    
    ### methods for revision
    def prepare_for_revision(self, input_name, n_examples_revision, include_true_answer):
        self._load_critiqued_dataset(input_name=input_name)
        
        if self.ready_for_revision == False:
            self.dataset = self.dataset.map(lambda x: self._template_for_revision(example=x, n_examples_revision=n_examples_revision, include_true_answer=include_true_answer))
            self.ready_for_revision = True

    def _load_critiqued_dataset(self, input_name):
        _n_shards = 10
        try:
            self.dataset = load_from_disk(f"../llm_outputs/rationales_critiqued/{input_name}")
        except FileNotFoundError:
            self.dataset = concatenate_datasets([load_from_disk(f"../llm_outputs/rationales_critiqued/{input_name}/shard_{shard_idx}") for shard_idx in range(_n_shards)])
        if self.sample < 1:
            logging.info("Loaded {sample} % of the dataset with critiques.".format(sample=int(self.sample*100)))
            self.dataset = self.dataset.train_test_split(train_size=self.sample)['train']    
        else:
            logging.info("Loaded 100 % of the data with critiques.")
    
    def _template_for_revision(self, example, n_examples_revision, include_true_answer):
        assert n_examples_revision > 0
        assert n_examples_revision < 8
        string_container = CritiqueStringContainer(dataset_name=self.dataset_name)
        
        few_shot_examples = "\n\n".join(string_container.get_random_revision_examples(n_examples_revision))
        critique = example['few_shot_critique'].split('\n')[0].strip()
        
        if include_true_answer == True:
            revision_string = f"{few_shot_examples}\n\nQuestion: {example['question']}\nAnswer: The answer is {self.answer_map[example['answer']]}. {example['few_shot_positive_rationale']}\nCritique: {critique}\n{example['few_shot_revision_instruction']}\nRevision: The answer is {self.answer_map[example['answer']]}."
        elif include_true_answer == False:
            revision_string = f"{few_shot_examples}\n\nQuestion: {example['question']}\nAnswer: The answer is {self.answer_map[example['answer']]}. {example['few_shot_positive_rationale']}\nCritique: {critique}\n{example['few_shot_revision_instruction']}\nRevision:"
        example['few_shot_revision_prompt'] = revision_string
        
        return example

    def prepare_to_cfrationalize(self, *args, **kwargs):
        self.dataset = self._load_from_huggingface()
        self.dataset = self.dataset.filter(lambda x: 'answer' in x and x['answer'] is not None)
        
        if self.ready_for_rationalizing == False:
            self.dataset = self.dataset.map(self._template_for_cfrationale, load_from_cache_file=False)
        self.ready_for_cfrationalizing = True
              
    
    def _template_for_cfrationale(self, example, *args, **kwargs):
        string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
        affixes = string_container.get_random_affixes(direction='positive')
        
        answers = ['yes', 'no']
        
        for i, answer in enumerate(answers):
            few_shot_example = string_container.get_few_shot_example()
            string = f"{few_shot_example}\n{example['question']}\nThe answer is {answer}.\nBriefly explain the reasoning behind this answer."
            col = f'counterfactual_prompt_answer_{i+1}'
            example[col] = string
        return example