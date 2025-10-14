from transformers import DataCollatorForLanguageModeling

class MultitaskDataCollator(DataCollatorForLanguageModeling):
    """Data collator for multitask decoder-only training."""
    
    def __init__(self, tokenizer, mlm=False, **kwargs):
        self.tasks = {"multitask_predict", "multitask_explain"}
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)

    def __call__(self, features, concat_inputs_and_labels=True):
        collated_task_inputs = {}
        for task in self.tasks:
            input_key = f'{task}_input_encoded_input_ids'
            label_key = f'{task}_label_encoded_input_ids'
            
            task_features = []
            for feature in features:
                inp = feature[input_key]
                lbl = feature[label_key]
                # Concatenate input and labels for decoder training
                task_features.append({'input_ids': inp + lbl if concat_inputs_and_labels else inp})
                
            collated_task_inputs[task] = super().__call__(task_features)
        
        return collated_task_inputs
    
    
class CounterfactualDataCollator(DataCollatorForLanguageModeling):
    """Data collator for multitask decoder-only training."""
    
    def __init__(self, tokenizer, mlm=False, **kwargs):
        self.tasks = {"correct_answer", "false_answer1"}
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)

    def __call__(self, features, concat_inputs_and_labels=True):
        collated_task_inputs = {}
        for task in self.tasks:
            input_key = f'{task}_input_encoded_input_ids'
            label_key = f'{task}_label_encoded_input_ids'
            
            task_features = []
            for feature in features:
                inp = feature[input_key]
                lbl = feature[label_key]
                # Concatenate input and labels for decoder training
                task_features.append({'input_ids': inp + lbl if concat_inputs_and_labels else inp})
                
            collated_task_inputs[task] = super().__call__(task_features)
        
        return collated_task_inputs


class CombinedDataCollator(DataCollatorForLanguageModeling):
    """Data collator for multitask decoder-only training."""
    
    def __init__(self, tokenizer, mlm=False, **kwargs):
        self.tasks = {"multitask_predict", "multitask_explain", "correct_answer", "false_answer1"}
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)

    def __call__(self, features, concat_inputs_and_labels=True):
        collated_task_inputs = {}
        for task in self.tasks:
            input_key = f'{task}_input_encoded_input_ids'
            label_key = f'{task}_label_encoded_input_ids'
            
            task_features = []
            for feature in features:
                inp = feature[input_key]
                lbl = feature[label_key]
                # Concatenate input and labels for decoder training
                task_features.append({'input_ids': inp + lbl if concat_inputs_and_labels else inp})
                
            collated_task_inputs[task] = super().__call__(task_features)
        
        return collated_task_inputs