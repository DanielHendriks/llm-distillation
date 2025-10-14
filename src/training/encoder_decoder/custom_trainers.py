import torch
import pandas as pd
from transformers import Seq2SeqTrainer
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn


class CounterfactualTrainer(Seq2SeqTrainer):
    """
    Trainer for counterfactual training of the student model.
    Based on Wang et al. (2023)
    """
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        correct_answer_outputs = model(**inputs['correct_answer'])
        false_answer1_outputs = model(**inputs['false_answer1'])

        loss = correct_answer_outputs.loss + false_answer1_outputs.loss
        return (loss, {'correct_answer': correct_answer_outputs, 'false_answer1': false_answer1_outputs}) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        correct_answer_outputs = super().prediction_step(model, inputs['correct_answer'], prediction_loss_only=False, ignore_keys=ignore_keys)
        
        if self.output_rationale:
            false_answer1_outputs = super().prediction_step(model, inputs['false_answer1'], prediction_loss_only=False, ignore_keys=ignore_keys)
        loss = correct_answer_outputs[0] + false_answer1_outputs[0] 
        
        return (
            loss,
            [correct_answer_outputs[1], false_answer1_outputs[1]],
            [correct_answer_outputs[2], false_answer1_outputs[2]],
        )
        


class MultitaskTrainer(Seq2SeqTrainer):
    """
    Trainer for multitask training of the student model.
    Based on Hsieh el al. (2023).
    """
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        if self.output_rationale:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs # placeholder only

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]
            
        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )
  

class CounterfactualAndMultitaskTrainer(Seq2SeqTrainer):
    """
    Trainer for combined counterfactual and multitask training of the student model.
    """
    def __init__(self, alpha, beta, gamma, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.output_rationale = output_rationale
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        # multitask loss
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])
        loss_multitask = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss
        
        # counterfactual loss
        correct_answer_outputs = model(**inputs['correct_answer'])
        false_answer1_outputs = model(**inputs['false_answer1'])

        loss_counterfactual = self.beta * correct_answer_outputs.loss + (1 - self.beta) * false_answer1_outputs.loss 

        # total loss
        loss = self.gamma * loss_multitask + (1 - self.gamma) * loss_counterfactual
        
        return (loss, 
                {
                    'pred': pred_outputs, 
                    'expl': expl_outputs, 
                    'correct_answer': correct_answer_outputs, 
                    'false_answer1': false_answer1_outputs, 
                 }
                ) if return_outputs else loss

    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        
        correct_answer_outputs = super().prediction_step(model, inputs['correct_answer'], prediction_loss_only=False, ignore_keys=ignore_keys)
        false_answer1_outputs = super().prediction_step(model, inputs['false_answer1'], prediction_loss_only=False, ignore_keys=ignore_keys)
        
        loss_counterfactual = self.beta * correct_answer_outputs[0] + (1 - self.beta) * false_answer1_outputs[0]
        loss_multitask = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]
        loss = self.gamma * loss_multitask + (1 - self.gamma) * loss_counterfactual

        return (
            loss,
            [pred_outputs[1], expl_outputs[1], correct_answer_outputs[1], false_answer1_outputs[1]],
            [pred_outputs[2], expl_outputs[2], correct_answer_outputs[2], false_answer1_outputs[2]],
        )
