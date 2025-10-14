import json
import logging
import os
from datasets import load_dataset, DatasetDict, load_from_disk, concatenate_datasets
from .generation_utils import RationaleGenerationStringContainer, CritiqueStringContainer
from abc import ABC, abstractmethod
from datasets import disable_caching
disable_caching()


class CQADatasetLoader:
    def __init__(self, sample=1) -> None:
        self.dataset_name = 'cqa'
        self.source_dataset_name = 'cos_e'
        self.dataset_version = 'v1.11'
        self.has_valid = False
        self.splits = ['train','validation']
        self.sample = sample
        
        # helper and assertion variables
        self.ready_for_rationalizing = False
        self.ready_for_critique = False
        self.ready_for_revision = False
        self.ready_for_cfrationalizing = False # cf = counterfactual
        self._seed = 42
    
    ### internal methods
    def _load_from_source(self):
        self.dataset = load_dataset(self.source_dataset_name, self.dataset_version)
    
    def _save_as_json(self):
        for key in self.dataset.keys():
            self.dataset[key].to_json("../datasets/cqa/{key}.json".format(key=key))
        
    
    ### methods for rationalizing
    def prepare_to_rationalize(self, prompting_mode: list, direction: list):
        self.dataset = self._load_from_json()
        # self.dataset = self._prepare_splits()
        
        if self.ready_for_rationalizing == False:
            # apply prompt template
            for mode in prompting_mode:
                for dire in direction:
                    self.dataset = self.dataset.map(lambda x: self._template_for_rationale(example=x, prompting_mode=mode, direction=dire))
            self.ready_for_rationalizing = True
            
    def _load_from_json(self):
        data_files = {f'{split}': f'../datasets/cqa/{split}.json' for split in self.splits}
        return load_dataset('json', data_files=data_files)
            
    def _template_for_rationale(self, example, prompting_mode, direction):
        # create choices string
        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        choices_str = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])
        
        # get random affixes for direction
        string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
        affixes = string_container.get_random_affixes(direction=direction)
        
        if prompting_mode == 'few_shot':
            few_shot_example = string_container.get_few_shot_example(affixes, direction)
            string_ = f"{few_shot_example}\nQuestion: {example['question']}\nChoices: {choices_str}\nAssistant ({affixes[0]} and {affixes[1]}): The answer is {example['answer']}."
        elif prompting_mode == 'zero_shot':
            string_ = f"Question: {example['question']}\nChoices: {choices_str}. The answer is {example['answer']}. \nAssistant ({affixes[0]} and {affixes[1]}): Let\'s think step by step. "
        elif prompting_mode == 'step-by-step':
            few_shot_example = string_container.get_few_shot_example(affixes, direction)
            string_ = f"{few_shot_example}\n\nQuestion: {example['question']}\nChoices: {choices_str}\nAssistant ({affixes[0]} and {affixes[1]}): "
            
        example[f'{prompting_mode}_{direction}_prompt'] = string_
        return example 
            

    ### methods for critiquing
    def prepare_for_critique(self, input_name, n_examples_critique, _n_shards = 10):
        # load dataset with critiques
        # split controlled trough input folder name
        self.dataset = self._load_rationalized_dataset(input_name=input_name)
        
        # prepare dataset for critique
        if self.ready_for_critique == False:
            # apply prompt template
            self.dataset = self.dataset.map(lambda x: self._template_for_critique(example=x, n_examples_critique=n_examples_critique))
            self.ready_for_critique = True
            logging.info("Dataset is ready for critique.")

    def _load_rationalized_dataset(self, input_name, _n_shards = 10):
        try: dataset = load_from_disk(f"../llm_outputs/rationales/{input_name}")
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
        csc = CritiqueStringContainer()
        assert len(csc.POSITIVE_RATIONALES) == len(csc.CRITIQUE_EXAMPLES) == len(csc.REVISION_EXAMPLES) == len(csc.FEW_SHOT_EXAMPLES_CRITIQUE_REQUEST) == len(csc.FEW_SHOT_EXAMPLES_REVISION_REQUEST)
        
        # create strings
        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        choices_str = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])
        few_shot_examples = "\n\n".join(csc.get_random_critique_examples(n_examples_critique))
        critique_request, revision_request = csc.get_random_critique_revision_request_pair()
        
        # create and store critique prompt and matching revision request string
        critique_string = f"{few_shot_examples}\n\nQuestion: {example['question']}\nChoices: {choices_str}\nAnswer: The answer is {example['answer']}. {example['few_shot_positive_rationale']}\n{critique_request}\nCritique: "
        example['few_shot_critique_prompt'] = critique_string
        example['few_shot_revision_instruction'] = revision_request
        
        return example
    
    
    ### methods for revision
    def prepare_for_revision(self, input_name, n_examples_revision, include_true_answer):
        # load dataset with critiques
        # split controlled trough input folder name
        self._load_critiqued_dataset(input_name=input_name)
        
        # prepare dataset for revision
        if self.ready_for_revision == False:
            # create revision prompt
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
        # check variables
        assert n_examples_revision > 0
        assert n_examples_revision < 8
        string_container = CritiqueStringContainer()
        
        # create strings
        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        choices = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])
        few_shot_examples = "\n\n".join(string_container.get_random_revision_examples(n_examples_revision)) #.replace(INSTRUCTION, revision_str)
        critique = example['few_shot_critique'].split('\n')[0].strip()
        
        # create and store revision request
        if include_true_answer == True:
            revision_string = f"{few_shot_examples}\n\nQuestion: {example['question']}\nChoices: {choices}\nAnswer: The answer is {example['answer']}. {example['few_shot_positive_rationale']}\nCritique: {critique}\n{example['few_shot_revision_instruction']}\nRevision: The answer is {example['answer']}."
        elif include_true_answer == False:
            revision_string = f"{few_shot_examples}\n\nQuestion: {example['question']}\nChoices: {choices}\nAnswer: The answer is {example['answer']}. {example['few_shot_positive_rationale']}\nCritique: {critique}\n{example['few_shot_revision_instruction']}\nRevision:"
        example['few_shot_revision_prompt'] = revision_string
        
        return example
    

    ### methods for counterfactual rationalizing 
    def prepare_to_cfrationalize(self):
        # prepare dataset for counterfactual rationale generation
        self.dataset = self._load_from_json()
        # self.dataset = self._prepare_splits() # 
        
        if self.ready_for_rationalizing == False:
            # apply prompt template
            self.dataset = self.dataset.map(self._template_for_cfrationale)
        self.ready_for_cfrationalizing = True
              
    
    def _template_for_cfrationale(self, example):
        # create choices string
        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        choices_str = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])
        
        string_container = RationaleGenerationStringContainer(dataset_name=self.dataset_name)
        affixes = string_container.get_random_affixes(direction='positive')
        
        for i, choice in enumerate(example['choices']):
            # generate string
            few_shot_example = string_container.get_few_shot_example(affixes, direction='positive')
            string = f"{few_shot_example}\nQuestion: {example['question']}\nChoices: {choices_str}\nAssistant ({affixes[0]} and {affixes[1]}): The answer is {choice}."
            col = f'counterfactual_prompt_answer_{i+1}'
            example[col] = string
        return example
    
    
    ### methods for training in general
    def prepare_for_training(self, path, model_type, tokenizer, 
                             max_length, target_rationale, 
                             truncation=True, padding=True,
                             do_tokenize = True):
        """
        model_type: should be either of the following:
            - "standard" prepare for standard finetuning. 
            - "task_prefix" prepare for multitask training with predicting the answer and explaining it. 
            - "counterfactual_prefix" prepare for counterfactual training to explain the true answer and predict the answer for false rationales.
            - "multitask_counterfactual_prefix" prepare for both multitaks and counterfatucal training simultaneously.
        
        Important: The dataset in path needs to contain specific columns depending on the model_type parameter! The dataset is expected to be stored in shards.
        """
        num_shards = 10
        dataset = concatenate_datasets([
            load_from_disk(f"{path}/shard_{shard_idx}")
            for shard_idx in range(num_shards)
        ])
        
        if model_type == "task_prefix":
            logging.debug("entered task prefix mode...")
            # apply template
            dataset = dataset.map(
                lambda x: self._template_for_taskprefix_training(
                    example=x, target_rationale=target_rationale,),
                load_from_cache_file=False,
                )
            
            if do_tokenize:
                # tokenize dataset
                tokenized_datasets = dataset.map(
                    lambda x: self._tokenize_function_taskprefix(
                        examples=x, tokenizer=tokenizer, 
                        max_length=max_length, truncation=truncation, 
                        padding=padding),
                    batched=True
                )
                # select relevant columns
                tokenized_datasets = tokenized_datasets.select_columns(
                    ['multitask_predict_input_encoded_input_ids', 
                    'multitask_predict_input_encoded_attention_mask', 
                    'multitask_predict_label_encoded_input_ids', 
                    'multitask_predict_label_encoded_attention_mask', 
                    'multitask_explain_input_encoded_input_ids', 
                    'multitask_explain_input_encoded_attention_mask', 
                    'multitask_explain_label_encoded_input_ids', 
                    'multitask_explain_label_encoded_attention_mask']
                )
            else:
                tokenized_datasets = dataset
            
        elif model_type == "counterfactual_prefix":
            logging.debug("entered counterfactual prefix mode...")
            # apply template
            dataset = dataset.map(lambda x: self._template_for_counterfactual_training(x, target_rationale=target_rationale))
            # tokenize dataset
            tokenized_datasets = dataset.map(
                lambda x: self._tokenize_function_counterfactual(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                batched=True
            )
            # select relevant columns
            #TODO
            
        elif model_type == "standard":
            logging.debug("entered standard mode...")
            # apply template
            dataset = dataset.map(self._template_for_stndrd_training)
            # tokenize dataset
            tokenized_datasets = dataset.map(
                lambda x: self._tokenize_function_standard(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                batched=True
            )
            # select relevant columns
            # dataset = dataset.select_columns(['input','label'])
            tokenized_datasets = tokenized_datasets.select_columns(
                ['input_ids','labels','attention_mask']
            )
        
        elif model_type == "both":  
            logging.debug("entered counterfactual plus multitask prefix mode...")
            # apply templates
            dataset = dataset.map(self._template_for_counterfactual_training)
            dataset = dataset.map(
                lambda x: self._template_for_taskprefix_training(example=x, target_rationale=target_rationale) 
                )
            # tokenize dataset
            tokenized_datasets = dataset.map(
                lambda x: self._tokenize_function_counterfactual(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                batched=True
            )
            tokenized_datasets = tokenized_datasets.map(
                lambda x: self._tokenize_function_taskprefix(examples=x, tokenizer=tokenizer, max_length=max_length, truncation=truncation, padding=padding),
                batched=True
            )
            # select relevant columns
            #TODO
                    
        else:
            raise ValueError
        
        return tokenized_datasets
    
    
    ### methods for multitask training
    def _template_for_taskprefix_training(self, example, target_rationale = None):
        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        choices_str = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])

        # when providing the [predict] tag, I want the model to output the answer to the question in the format "The answer is XYZ".
        example['multitask_predict_input'] = f"[predict] Question: {example['question']}\nChoices: {choices_str} "
        example['multitask_predict_label'] = f"[answer] The answer is {example['answer']}"
        # when providing the [explain] tag, I want the model to output the rationale.
        example['multitask_explain_input'] = f"[explain] Question: {example['question']}\nChoices: {choices_str} " 
        
        if target_rationale is not None: 
            example['multitask_explain_label'] = example[target_rationale].split('\n')[0].strip()
        return example
    
    def _tokenize_function_taskprefix(self, examples, tokenizer, max_length, truncation=True, padding=True):
        model_inputs = {
            "multitask_predict_input_encoded": tokenizer(examples["multitask_predict_input"], max_length=max_length, truncation=truncation, padding=padding),
            "multitask_predict_label_encoded": tokenizer(examples["multitask_predict_label"], max_length=300, truncation=truncation, padding=padding),
            "multitask_explain_input_encoded": tokenizer(examples["multitask_explain_input"], max_length=max_length, truncation=truncation, padding=padding),
            "multitask_explain_label_encoded": tokenizer(examples["multitask_explain_label"], max_length=300, truncation=truncation, padding=padding),
            }
        
        model_inputs_flattened = {}
        for key_outer in model_inputs:
            for key_inner in model_inputs[key_outer]:
                model_inputs_flattened[f"{key_outer}_{key_inner}"] = model_inputs[key_outer][key_inner]

        return model_inputs_flattened
    
    
    ### methods for counterfactual training 
    def _template_for_counterfactual_training(self, example, target_rationale=None):
        if target_rationale is not None:
            _correct_rationale_name = target_rationale
        if target_rationale is None:
            _correct_answer_index = example["choices"].index(example["answer"])
            _correct_rationale_name = f"counterfactual_prompt_answer_{_correct_answer_index + 1}_rationale" 
            
        _cf_answer_cols = [f"counterfactual_prompt_answer_{i}_rationale" for i in range(1, 6)]

        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        _choices_str = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])

        # when providing the [factual] tag, I want the model to output the corresponding rationale 
        # and subsequently the correct answer and in the format "So the answer is XYZ".
        example["correct_answer_input"] = f"[factual] Question: {example['question']}\nChoices: {_choices_str} "
        first_line = example[_correct_rationale_name].split('\n')[0].strip()
        example["correct_answer_label"] = f"[factual] Answer: {first_line} So the answer is {example['answer']}." # \nAnswer: The answer is {example['answer']}."
                                        # previously: Rationale:
                                            
        count = 1
        for i, col_name in enumerate(_cf_answer_cols):
            if col_name != _correct_rationale_name:
                # when providing the [counterfactual] tag, I want the model to output the rationale the the corresponding provided false answer.
                rational = example[col_name].split('\n')[0].strip()
                example[f"false_answer{count}_input"] = f"[counterfactual] Question: {example['question']}\nChoices: {_choices_str}\Rationale: {rational} "
                example[f"false_answer{count}_label"] = f"[counterfactual] So the answer is {example['choices'][i]}." # The answer is {example['choices'][i]}."
                count += 1
        return example
    
    def _tokenize_function_counterfactual(self, examples, tokenizer, max_length, truncation=True, padding=True):
        model_inputs = {
            'correct_answer_input_encoded': tokenizer(examples['correct_answer_input'], max_length=max_length, truncation=truncation, padding=padding),
            'correct_answer_label_encoded': tokenizer(examples['correct_answer_label'], max_length=256, truncation=truncation, padding=padding),
            'false_answer1_input_encoded': tokenizer(examples['false_answer1_input'], max_length=max_length, truncation=truncation, padding=padding),
            'false_answer1_label_encoded': tokenizer(examples['false_answer1_label'], max_length=256, truncation=truncation, padding=padding),
            'false_answer2_input_encoded': tokenizer(examples['false_answer2_input'], max_length=max_length, truncation=truncation, padding=padding),
            'false_answer2_label_encoded': tokenizer(examples['false_answer2_label'], max_length=256, truncation=truncation, padding=padding),
            'false_answer3_input_encoded': tokenizer(examples['false_answer3_input'], max_length=max_length, truncation=truncation, padding=padding),
            'false_answer3_label_encoded': tokenizer(examples['false_answer3_label'], max_length=256, truncation=truncation, padding=padding),
            'false_answer4_input_encoded': tokenizer(examples['false_answer4_input'], max_length=max_length, truncation=truncation, padding=padding),
            'false_answer4_label_encoded': tokenizer(examples['false_answer4_label'], max_length=256, truncation=truncation, padding=padding),
        }
            
        model_inputs_flattened = {}

        for key_outer in model_inputs:
            for key_inner in model_inputs[key_outer]:
                model_inputs_flattened[f"{key_outer}_{key_inner}"] = model_inputs[key_outer][key_inner]

        return model_inputs_flattened


    ### methods for standard training
    def _template_for_stndrd_training(self, example):
        #TODO
        _letter_map = {0: '(a)',  1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)', 6: '(g)'}
        choices_str = ' '.join([f'{_letter_map[i]} {c}' for i, c in enumerate(example['choices'])])
        example['input'] = f"[question] {example['question']}\n[choices] {choices_str}"
        example['label'] = f"[answer] The answer is {example['answer']}. {example['few_shot_positive_rationale']}"
        
        return example
    
    def _tokenize_function_standard(self, examples, tokenizer, max_length, truncation=True, padding=True):
        model_inputs = tokenizer(
            examples['input'],
            max_length=max_length,
            truncation=truncation,
            padding=padding
        )

        label_output_encodings = tokenizer(text_target=examples['label'], max_length=256, truncation=truncation, padding=padding)
        model_inputs['labels'] = label_output_encodings['input_ids']

        return model_inputs
    
    

    