import torch
import pandas as pd
from transformers import DataCollatorForSeq2Seq
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn


"""T5 Multi-Task"""
class MultitaskDataCollator(DataCollatorForSeq2Seq):
    """
    Based on the paper "Distillation step-by-step" by Hsieh el al. (2023).
    """
    
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)

        pred_features = features_df.loc[:, features_df.columns.isin(['multitask_predict_label_encoded_input_ids', 'multitask_predict_input_encoded_input_ids', 'multitask_predict_input_encoded_attention_mask'])] \
            .rename(columns={'multitask_predict_label_encoded_input_ids': 'labels', 'multitask_predict_input_encoded_input_ids': 'input_ids', 'multitask_predict_input_encoded_attention_mask': 'attention_mask'}) \
            .to_dict('records')

        expl_features = features_df.loc[:, features_df.columns.isin(['multitask_explain_label_encoded_input_ids', 'multitask_explain_input_encoded_input_ids', 'multitask_explain_input_encoded_attention_mask'])] \
            .rename(columns={'multitask_explain_label_encoded_input_ids': 'labels', 'multitask_explain_input_encoded_input_ids': 'input_ids', 'multitask_explain_input_encoded_attention_mask': 'attention_mask'}) \
            .to_dict('records')
        
        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }



"""T5 Counterfactual"""
class CounterfactualDataCollator(DataCollatorForSeq2Seq):
    """
    Based on the paper "Self-Consistent Chain-of-Thought Distillation" from Wang el al. (2023).
    """
    
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        correct_answer_features = features_df.loc[:, features_df.columns.isin(['correct_answer_label_encoded_input_ids', 'correct_answer_input_encoded_input_ids', 'correct_answer_input_encoded_attention_mask'])].rename(
            columns={'correct_answer_label_encoded_input_ids': 'labels', 'correct_answer_input_encoded_input_ids': 'input_ids', 'correct_answer_input_encoded_attention_mask': 'attention_mask'}).to_dict('records')
        
        false_answer1_features = features_df.loc[:, features_df.columns.isin(['false_answer1_label_encoded_input_ids', 'false_answer1_input_encoded_input_ids', 'false_answer1_input_encoded_attention_mask'])].rename(
            columns={'false_answer1_label_encoded_input_ids': 'labels', 'false_answer1_input_encoded_input_ids': 'input_ids', 'false_answer1_input_encoded_attention_mask': 'attention_mask'}).to_dict('records')
        
        correct_answer_features = super().__call__(correct_answer_features, return_tensors)
        false_answer1_features = super().__call__(false_answer1_features, return_tensors)

        return {
            'correct_answer': correct_answer_features,
            'false_answer1': false_answer1_features,
        }
        

"""T5 Counterfactual and Multitask simultaneous"""
class CounterfactualAndMultitaskDataCollator(DataCollatorForSeq2Seq): 
    def __call__(self, features, return_tensors=None):
        
        output_taskprefix_datacollator = MultitaskDataCollator(self.tokenizer)(features, return_tensors)
        output_couterfactual_datacollator = CounterfactualDataCollator(self.tokenizer)(features, return_tensors)
        
        return {
            **output_taskprefix_datacollator, 
            **output_couterfactual_datacollator
            }
