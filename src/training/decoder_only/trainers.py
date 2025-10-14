from transformers import Trainer
import torch

class MultitaskTrainer(Trainer):
    """Custom trainer for multitask decoder-only training."""
    
    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        self.tasks = ["multitask_predict", "multitask_explain"]
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_outputs = {}    
        for task in self.tasks:
            task_outputs[task] = model(**inputs[task])
        
        predict_loss = task_outputs["multitask_predict"].loss
        explain_loss = task_outputs["multitask_explain"].loss
        combined_loss = self.alpha * predict_loss + (1. - self.alpha) * explain_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train_predict_loss': predict_loss.item(),
                'train_explain_loss': explain_loss.item(),
                'train_combined_loss': combined_loss.item(),
                'alpha': self.alpha
            })

        return (combined_loss, task_outputs) if return_outputs else combined_loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            task_outputs = {}    
            for task in self.tasks:
                task_outputs[task] = model(**inputs[task])
        
        predict_loss = task_outputs["multitask_predict"].loss
        explain_loss = task_outputs["multitask_explain"].loss
        combined_loss = self.alpha * predict_loss + (1. - self.alpha) * explain_loss

        if prediction_loss_only:
            combined_logits = None
        else:
            combined_logits = {
                'predict': task_outputs["multitask_predict"].logits if hasattr(task_outputs["multitask_predict"], 'logits') else None,
                'explain': task_outputs["multitask_explain"].logits if hasattr(task_outputs["multitask_explain"], 'logits') else None
            }
        
        return combined_loss, combined_logits, None


class CounterfactualTrainer(Trainer):
    """Custom trainer for counterfactual decoder-only training."""
    
    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        self.tasks = {"correct_answer", "false_answer1"}
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_outputs = {}    
        for task in self.tasks:
            task_outputs[task] = model(**inputs[task])
        
        factual_loss = task_outputs["correct_answer"].loss
        counterfactual_loss = task_outputs["false_answer1"].loss
        combined_loss = self.alpha * factual_loss + (1. - self.alpha) * counterfactual_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train_factual_loss': factual_loss.item(),
                'train_counterfactual_loss': counterfactual_loss.item(),
                'train_combined_loss': combined_loss.item(),
                'alpha': self.alpha
            })

        return (combined_loss, task_outputs) if return_outputs else combined_loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            task_outputs = {}    
            for task in self.tasks:
                task_outputs[task] = model(**inputs[task])
        
        predict_loss = task_outputs["correct_answer"].loss
        explain_loss = task_outputs["false_answer1"].loss
        combined_loss = self.alpha * predict_loss + (1. - self.alpha) * explain_loss

        if prediction_loss_only:
            combined_logits = None
        else:
            combined_logits = {
                'correct_answer': task_outputs["correct_answer"].logits if hasattr(task_outputs["correct_answer"], 'logits') else None,
                'counterfactual': task_outputs["false_answer1"].logits if hasattr(task_outputs["false_answer1"], 'logits') else None
            }
        
        return combined_loss, combined_logits, None


class CombinedTrainer(Trainer):
    """Custom trainer for combined decoder-only training."""
    
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tasks = {"multitask_predict", "multitask_explain", "correct_answer", "false_answer1"}
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_outputs = {}    
        for task in self.tasks:
            task_outputs[task] = model(**inputs[task])
        
        predict_loss = task_outputs["multitask_predict"].loss
        explain_loss = task_outputs["multitask_explain"].loss
        factual_loss = task_outputs["correct_answer"].loss
        counterfactual_loss = task_outputs["false_answer1"].loss
        
        mt_loss = self.beta * predict_loss + (1. - self.beta) * explain_loss
        cf_loss = self.gamma * factual_loss + (1. - self.gamma) * counterfactual_loss
        combined_loss = self.alpha * mt_loss + (1. - self.alpha) * cf_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train_mt_loss': mt_loss.item(),
                'train_cf_loss': cf_loss.item(),
                'train_combined_loss': combined_loss.item(),
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
            })

        return (combined_loss, task_outputs) if return_outputs else combined_loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            task_outputs = {}    
            for task in self.tasks:
                task_outputs[task] = model(**inputs[task])
        
        predict_loss = task_outputs["multitask_predict"].loss
        explain_loss = task_outputs["multitask_explain"].loss
        factual_loss = task_outputs["correct_answer"].loss
        counterfactual_loss = task_outputs["false_answer1"].loss
        
        mt_loss = self.beta * predict_loss + (1. - self.beta) * explain_loss
        cf_loss = self.gamma * factual_loss + (1. - self.gamma) * counterfactual_loss
        combined_loss = self.alpha * mt_loss + (1. - self.alpha) * cf_loss

        if prediction_loss_only:
            combined_logits = None
        else:
            combined_logits = {
                'predict': task_outputs["multitask_predict"].logits if hasattr(task_outputs["multitask_predict"], 'logits') else None,
                'explain': task_outputs["multitask_explain"].logits if hasattr(task_outputs["multitask_explain"], 'logits') else None,
                'correct_answer': task_outputs["correct_answer"].logits if hasattr(task_outputs["correct_answer"], 'logits') else None,
                'counterfactual': task_outputs["false_answer1"].logits if hasattr(task_outputs["false_answer1"], 'logits') else None
            }
        
        return combined_loss, combined_logits, None
