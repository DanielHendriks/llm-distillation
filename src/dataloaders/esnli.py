import logging
from datasets import load_dataset, load_from_disk, concatenate_datasets
from .generation_utils import RationaleGenerationStringContainer, CritiqueStringContainer
from datasets import disable_caching
disable_caching()


class ESNLIDatasetLoader:
    """
    ESNLI Dataset Loader for Natural Language Inference with explanations.
    
    Similar to CQADatasetLoader but adapted for NLI task structure with premise/hypothesis pairs
    and entailment/neutral/contradiction labels. Features hybrid loading that tries Arrow format
    first and falls back to HuggingFace if not available.
    """
    
    def __init__(self, sample=1) -> None:
        # Dataset specific properties
        self.dataset_name = 'esnli'
        self.source_dataset_name = "stanfordnlp/snli"
        self.dataset_version = None  # Uses default version
        self.has_valid = True  # ESNLI has train/validation/test splits
        
        # Label mapping for NLI task
        self.label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        self.letter_map = {0: '(a)', 1: '(b)', 2: '(c)'}
        
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

    def _load_from_huggingface(self, split='train'):
        """Load ESNLI dataset from HuggingFace as fallback"""
        ds = load_dataset(self.source_dataset_name)
        
        # Randomly sample 10% of train and validation splits with fixed seed
        for sample_proportion, split_name in zip([50,10],['train', 'validation']):
            if split_name in ds:
                ds[split_name] = ds[split_name].shuffle(seed=42).select(range(len(ds[split_name]) // sample_proportion))
        
        ds = ds.filter(lambda x: x['label'] in self.label_map)
        
        return ds
    
    def _infer_split_from_path(self, path):
        """Infer which split to load from HuggingFace based on path"""
        path_lower = str(path).lower()
        if 'test' in path_lower:
            return 'test'
        elif 'val' in path_lower or 'validation' in path_lower:
            return 'validation'
        else:
            return 'train'
    
    def _get_safe_rationale(self, example, target_rationale):
        """Safely get rationale with fallback logic"""
        # Try target rationale first
        if target_rationale in example and example[target_rationale]:
            return example[target_rationale]
    
    ### Main training preparation method (CQA-compatible interface)
    def prepare_for_training(self, path, model_type, tokenizer, max_length, target_rationale, do_tokenize=True, truncation=True, padding=True):
        """
        Prepare ESNLI dataset for training with hybrid loading.
        
        Args:
            path: Path to preprocessed Arrow format data
            model_type: Training mode - currently supports 'task_prefix'
            tokenizer: HuggingFace tokenizer for encoding
            max_length: Maximum sequence length
            target_rationale: Which rationale to use ('explanation_1', 'llama_rationale', etc.)
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

        # Filter out examples with invalid labels
        dataset = dataset.filter(lambda x: x['label'] in self.label_map.keys())
        
        # Apply appropriate template based on model type
        if model_type == "task_prefix":
            logging.debug("Entered task prefix mode...")
            # Add this before processing to understand your data better:
            label_counts = {}
            for example in dataset:
                label = example['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print("Label distribution:", label_counts)
            print("Available label_map keys:", list(self.label_map.keys()))

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
                        examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
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
            raise NotImplementedError(f"Model type '{model_type}' not yet implemented for ESNLI")
        
        return tokenized_datasets
    
    ### Template methods for task prefix training
    def _template_for_taskprefix_training(self, example, target_rationale='explanation_1'):
        """
        Template ESNLI data for task_prefix training mode.
        
        Creates multitask format with [predict] and [explain] prefixes compatible with
        existing training pipeline. Supports flexible rationale selection.
        """
        # Create choices string in the format expected by the pipeline
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                               for i, label in enumerate(['entailment', 'neutral', 'contradiction'])])
        
        # Create question format for NLI (matches prototype findings)
        question = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']} What is the relationship?"
        
        # Prediction task: model should output "The answer is [label]"
        example['multitask_predict_input'] = f"[predict] Question: {question}\nChoices: {choices_str}"
        example['multitask_predict_label'] = f"The answer is {self.label_map[example['label']]}"
        
        # Explanation task: model should output the rationale
        example['multitask_explain_input'] = f"[explain] Question: {question}\nChoices: {choices_str}"
        
        # Use flexible rationale selection with fallback
        example['multitask_explain_label'] = self._get_safe_rationale(example, target_rationale).split('\n')[0].strip()
        
        return example
    
    def _tokenize_function_taskprefix(self, examples, tokenizer, max_length, truncation=True,padding=True):
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
        Template ESNLI data for standard training mode.
        
        Creates simple question/answer format compatible with standard fine-tuning.
        """
        # Create choices string in the format expected by the pipeline
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                               for i, label in enumerate(['entailment', 'neutral', 'contradiction'])])
        
        # Create question format for NLI
        question = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']} What is the relationship?"
        
        example['input'] = f"[question] {question}\n[choices] {choices_str}"
        
        # Use flexible rationale selection with fallback
        rationale = self._get_safe_rationale(example, target_rationale) if target_rationale else ''
        example['label'] = f"[answer] The answer is {self.label_map[example['label']]}. {rationale}"
        
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

        label_output_encodings = tokenizer(text_target=examples['label'], max_length=256, truncation=truncation)
        model_inputs['labels'] = label_output_encodings['input_ids']

        return model_inputs
    
    ### Template methods for counterfactual training
    def _template_for_counterfactual_training(self, example, target_rationale=None):
        if target_rationale is not None:
            _correct_rationale_name = target_rationale
        else:
            raise ValueError("Please specify target_rationale for counterfactual training.")
            
        # Create choices string
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                            for i, label in enumerate(['entailment', 'neutral', 'contradiction'])])
        
        # Create question format
        question = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']} What is the relationship?"
        
        # Factual case: correct answer with correct rationale
        example["correct_answer_input"] = f"[factual] Question: {question}\nChoices: {choices_str}"
        correct_rationale = self._get_safe_rationale(example, _correct_rationale_name).split('\n')[0].strip()
        
        first_line = correct_rationale.split("\n")[0].strip()
        example["correct_answer_label"] = f"[factual] Answer: {first_line} So the answer is {self.label_map[example['label']]}."
        
        # Counterfactual cases: incorrect answers with generated rationales
        labels = ['entailment', 'neutral', 'contradiction']
        count = 1
        for idx in range(1, 4):  # dataset keys use 1–3
            cf_rationale_col = f"counterfactual_prompt_answer_{idx}_rationale"
            if cf_rationale_col in example and example[cf_rationale_col]:
                label = labels[idx-1]  # shift back to 0-based
                if labels.index(label) != example['label']:  # skip correct one
                    rational = example[cf_rationale_col].split('\n')[0].strip()
                    example[f"false_answer{count}_input"] = f"[counterfactual] Question: {question}\nChoices: {choices_str}\nRationale: {rational}"
                    example[f"false_answer{count}_label"] = f"[counterfactual] So the answer is {label}."
                    count += 1
                    
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
        
        # Add false answers if they exist
        for i in range(1, 3):  # Up to 2 false answers for NLI (3 labels - 1 correct)
            false_input_key = f'false_answer{i}_input'
            false_label_key = f'false_answer{i}_label'
            
            if false_input_key in examples and examples[false_input_key]:
                model_inputs[f'false_answer{i}_input_encoded'] = tokenizer(examples[false_input_key], max_length=max_length, truncation=truncation, padding=padding)
                model_inputs[f'false_answer{i}_label_encoded'] = tokenizer(examples[false_label_key], max_length=256, truncation=truncation, padding=padding)
        
        # Flatten the tokenized dictionaries
        model_inputs_flattened = {}
        for key_outer in model_inputs:
            for key_inner in model_inputs[key_outer]:
                model_inputs_flattened[f"{key_outer}_{key_inner}"] = model_inputs[key_outer][key_inner]

        return model_inputs_flattened


    ### methods for rationalizing
    def prepare_to_rationalize(self, prompting_mode: list, direction: list):
        self.dataset = self._load_from_huggingface()
    
        if not self.ready_for_rationalizing:
            string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
            few_shot_prefix = string_container.get_few_shot_example()
    
            def add_prompts(batch):
                out = {}
                questions = [f"Premise: {p}\nHypothesis: {h} What is the relationship?" 
                             for p, h in zip(batch["premise"], batch["hypothesis"])]
                for mode in prompting_mode:
                    for dire in direction:
                        texts = []
                        for q, label in zip(questions, batch["label"]):
                            text = ""
                            if mode == 'few_shot':
                                text += few_shot_prefix + "\n"
                            text += f"{q}\nThe answer is {self.label_map[label]}."
                            text += "\nBriefly explain why this answer is the best fit to the relationship between premise and hypothesis."
                            texts.append(text)
                        out[f"{mode}_{dire}_prompt"] = texts
                return out
    
            self.dataset = self.dataset.map(add_prompts, batched=True, batch_size=1000)
            self.ready_for_rationalizing = True

            
    def _load_from_json(self):
        data_files = {f'{split}': f'../datasets/cqa/{split}.json' for split in self.splits}
        return load_dataset('json', data_files=data_files)
            
    def _template_for_rationale(self, example, prompting_mode, direction):
        # Create choices string in the format expected by the pipeline
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                               for i, label in enumerate(['entailment', 'neutral', 'contradiction'])])
        
        # Create question format for NLI (matches prototype findings)
        question = f"""Premise: {example['premise']} 
Hypothesis: {example['hypothesis']} What is the relationship?"""
        
        # get random affixes for direction
        string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
        # affixes = string_container.get_random_affixes(direction=direction)

        text = ""
        if prompting_mode == 'few_shot':
            text += string_container.get_few_shot_example()
            text += "\n"
        text += f"{question}\nThe answer is {self.label_map[example['label']]}."
        text += "\nBriefly explain why this answer is the best fit to the relationship between premise and hypothesis."
        
        example[f'{prompting_mode}_{direction}_prompt'] = text
        return example 
    
    
    ### methods for critiquing
    def prepare_for_critique(self, input_name, n_examples_critique, _n_shards = 10):
        self.dataset = self._load_rationalized_dataset(input_name=input_name, _n_shards = _n_shards)
        
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
        
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                               for i, label in enumerate(['entailment', 'neutral', 'contradiction'])])
        few_shot_examples = "\n\n".join(csc.get_random_critique_examples(n_examples_critique))
        critique_request, revision_request = csc.get_random_critique_revision_request_pair()
        
        critique_string = f"{few_shot_examples}\n\nQuestion: Premise: {example['premise']} Hypothesis: {example['hypothesis']} What is the relationship?\nChoices: {choices_str}\nAnswer: The answer is {self.label_map[example['label']]}. {example['few_shot_positive_rationale']}\n{critique_request}\nCritique: "
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
        
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                               for i, label in enumerate(['entailment', 'neutral', 'contradiction'])])
        few_shot_examples = "\n\n".join(string_container.get_random_revision_examples(n_examples_revision))
        critique = example['few_shot_critique'].split('\n')[0].strip()
        
        if include_true_answer == True:
            revision_string = f"{few_shot_examples}\n\nQuestion: Premise: {example['premise']} Hypothesis: {example['hypothesis']} What is the relationship?\nChoices: {choices_str}\nAnswer: The answer is {self.label_map[example['label']]}. {example['few_shot_positive_rationale']}\nCritique: {critique}\n{example['few_shot_revision_instruction']}\nRevision: The answer is {self.label_map[example['label']]}."
        elif include_true_answer == False:
            revision_string = f"{few_shot_examples}\n\nQuestion: Premise: {example['premise']} Hypothesis: {example['hypothesis']} What is the relationship?\nChoices: {choices_str}\nAnswer: The answer is {self.label_map[example['label']]}. {example['few_shot_positive_rationale']}\nCritique: {critique}\n{example['few_shot_revision_instruction']}\nRevision:"
        example['few_shot_revision_prompt'] = revision_string
        
        return example

    def prepare_to_cfrationalize(self):
        self.dataset = self._load_from_huggingface()
        
        if self.ready_for_rationalizing == False:
            self.dataset = self.dataset.map(self._template_for_cfrationale)
        self.ready_for_cfrationalizing = True
              
    
    def _template_for_cfrationale(self, example):
        string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
        affixes = string_container.get_random_affixes(direction='positive')
        
        labels = ['entailment', 'neutral', 'contradiction']
        choices_str = ' '.join([f'{self.letter_map[i]} {label}' 
                               for i, label in enumerate(labels)])
        
        for i, label in enumerate(labels):
            few_shot_example = string_container.get_few_shot_example()
            string = f"{few_shot_example}\nPremise: {example['premise']} \nHypothesis: {example['hypothesis']} What is the relationship?\nThe answer is {label}. \nBriefly explain why this answer is the best fit to the relationship between premise and hypothesis."
            col = f'counterfactual_prompt_answer_{i+1}'
            example[col] = string
        return example



