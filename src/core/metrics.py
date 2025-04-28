import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
import sacrebleu
from collections import defaultdict


class Metrics:
    def __init__(self, metrics_kwargs):
        self.metrics = {}
        # Store the metrics configuration
        self.metrics_kwargs = metrics_kwargs
        self.batch_metrics = defaultdict(list)
        # Group metrics by the type of prediction they require
        self.token_metrics = ['BLEU', 'ROUGE', 'METEOR', 'SacreBLEU', 'SacreBLEU-detok', 
                             'SacreBLEU-detok-smooth', 'SacreBLEU-detok-smooth-tokenized', 'CIDEr']
        self.logit_metrics = ['perplexity']
        self.classification_metrics = ['accuracy']
        self.regression_metrics = ['mse', 'mae', 'r2']

    def metric_lookup(self, metric_name):
        if metric_name == 'mse':
            return torch.nn.MSELoss()
        elif metric_name == 'mae':
            return torch.nn.L1Loss()
        elif metric_name == 'r2':
            return torch.nn.R2Loss()
        elif metric_name == 'accuracy':
            return self.accuracy
        elif metric_name == 'BLEU':
            return self.bleu_score
        # elif metric_name == 'ROUGE':
        #     return self.rouge_score
        elif metric_name == 'METEOR':
            return self.meteor_score
        elif metric_name == 'SacreBLEU':
            return self.sacrebleu_score
        elif metric_name == 'SacreBLEU-detok':
            return lambda y_true, y_pred: self.sacrebleu_score(y_true, y_pred, tokenize='none')
        elif metric_name == 'SacreBLEU-detok-smooth':
            return lambda y_true, y_pred: self.sacrebleu_score(y_true, y_pred, tokenize='none', smooth_method='exp')
        elif metric_name == 'SacreBLEU-detok-smooth-tokenized':
            return lambda y_true, y_pred: self.sacrebleu_score(y_true, y_pred, tokenize='13a', smooth_method='exp')
        elif metric_name == 'CIDEr':
            return self.cider_score
        elif metric_name == 'perplexity':
            return self.perplexity

    def accuracy(self, y_true, y_pred):
        """
        Calculate accuracy for token prediction tasks
        
        Args:
            y_true: Ground truth token IDs (batch_size, seq_len)
            y_pred: Either predicted token IDs (batch_size, seq_len) 
                   or logits (batch_size, seq_len, vocab_size)
        """
        # Check if y_pred contains logits or token IDs
        if len(y_pred.shape) == 3 and y_pred.size(-1) > 1:
            # This is logits tensor (batch_size, seq_len, vocab_size)
            pred_classes = torch.argmax(y_pred, dim=-1)  # Get token IDs from logits
        else:
            # This is already token IDs
            pred_classes = y_pred
            
        # Make sure both have the same shape
        min_len = min(pred_classes.size(1), y_true.size(1))
        pred_classes = pred_classes[:, :min_len]
        y_true = y_true[:, :min_len]
            
        # Calculate token-level accuracy
        correct = (pred_classes == y_true).float()
        
        # Skip padding tokens if present (usually token_id 0)
        mask = (y_true != 0).float()
        if mask.sum() > 0:
            return (correct * mask).sum() / mask.sum()
        else:
            return correct.mean()
    
    def bleu_score(self, y_true, y_pred):
        """
        Calculate BLEU score for text generation tasks
        
        Args:
            y_true: List of reference sentences (tokenized lists)
            y_pred: List of candidate sentences (tokenized lists)
        """
        if isinstance(y_true[0], torch.Tensor):
            y_true = [t.cpu().numpy().tolist() for t in y_true]
        if isinstance(y_pred[0], torch.Tensor):
            y_pred = [p.cpu().numpy().tolist() for p in y_pred]
            
        # Convert references to required format [[ref]] for each candidate
        references = [[ref] for ref in y_true]
        return corpus_bleu(references, y_pred)
    
    # def rouge_score(self, y_true, y_pred):
    #     """
    #     Calculate ROUGE score for text generation tasks
        
    #     Args:
    #         y_true: List of reference sentences (strings)
    #         y_pred: List of candidate sentences (strings)
    #     """
    #     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    #     scores = [scorer.score(ref, pred) for ref, pred in zip(y_true, y_pred)]
        
    #     # Return average rougeL F1 score across all samples
    #     return np.mean([score['rougeL'].fmeasure for score in scores])
    
    def meteor_score(self, y_true, y_pred):
        """
        Calculate METEOR score for text generation tasks
        
        Args:
            y_true: List of reference sentences (strings)
            y_pred: List of candidate sentences (strings)
        """
        scores = [meteor_score([ref], pred) for ref, pred in zip(y_true, y_pred)]
        return np.mean(scores)
    
    def sacrebleu_score(self, y_true, y_pred, tokenize='13a', smooth_method='none'):
        """
        Calculate SacreBLEU score for text generation tasks
        
        Args:
            y_true: List of reference sentences (strings)
            y_pred: List of candidate sentences (strings)
            tokenize: Tokenization method ('13a', 'intl', 'char', 'none')
            smooth_method: Smoothing method ('none', 'floor', 'add-k', 'exp')
        """
        # Convert to list of strings if needed
        if isinstance(y_true[0], list) or isinstance(y_true[0], torch.Tensor):
            y_true = [' '.join(map(str, ref)) for ref in y_true]
        if isinstance(y_pred[0], list) or isinstance(y_pred[0], torch.Tensor):
            y_pred = [' '.join(map(str, pred)) for pred in y_pred]
            
        # SacreBLEU expects references as a list of lists (multiple refs per example)
        references = [[ref] for ref in y_true]
        
        # Calculate score for each prediction
        scores = []
        for i, pred in enumerate(y_pred):
            refs = [references[i][0]]  # Get the reference for this prediction
            bleu = sacrebleu.corpus_bleu(
                [pred], 
                [refs],
                tokenize=tokenize,
                smooth_method=smooth_method
            )
            scores.append(bleu.score)
        
        return np.mean(scores)
    
    def cider_score(self, y_true, y_pred):
        """
        Simplified CIDEr score implementation
        
        Args:
            y_true: List of reference sentences (strings)
            y_pred: List of candidate sentences (strings)
        """
        # This is a simplified implementation; a full implementation would use n-grams and TF-IDF weighting
        return self.bleu_score(y_true, y_pred)  # Fallback to BLEU for simplicity
    
    def perplexity(self, y_true, y_pred):
        """
        Calculate perplexity for language models
        
        Args:
            y_true: Target token IDs (batch_size, seq_len)
            y_pred: Predicted logits (batch_size, seq_len, vocab_size)
        """
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
        return torch.exp(loss)

    def get_metric(self, metric_name):
        return self.metric_lookup(metric_name)
    
    def calculate_metrics(self, y_true, model_outputs):
        """
        Calculate metrics for all tasks based on metrics_kwargs
        
        Args:
            y_true: Dictionary of ground truth values by task
            model_outputs: Dictionary of model outputs by task
            
        Returns:
            Dictionary of all calculated metrics
        """
        all_results = {}
        
        # Calculate metrics for each task in metrics_kwargs
        for task in self.metrics_kwargs:
            if task in y_true:
                task_results = self._calculate_metric(y_true, model_outputs, self.metrics_kwargs[task])
                all_results.update(task_results)
        
        return all_results

    def _calculate_metric(self, y_true, model_outputs, metrics_list=None):
        """
        Calculate metrics based on the type of output required
        
        Args:
            y_true: Dictionary of ground truth values
            model_outputs: Dictionary of model outputs containing tokens and logits
            metrics_list: List of metrics to calculate for this task
            
        Returns:
            Dictionary of metrics with task-prefixed keys (e.g., 'title_accuracy')
        """
        results = {}
        
        # Get task name from y_true (e.g., 'title')
        for task in y_true:
            # Skip if no metrics defined for this task
            if task not in self.metrics_kwargs:
                continue
                
            # Get metrics list for this task
            task_metrics = self.metrics_kwargs[task].get('metrics', [])
            
            for metric_name in task_metrics:
                metric = self.get_metric(metric_name)
                metric_key = f"{task}_{metric_name}"
                
                # Determine which output to use based on metric type
                if metric_name in self.token_metrics:
                    # For BLEU, ROUGE, etc. use the token IDs
                    y_pred = model_outputs.get(task, None)
                    result = metric(y_true[task], y_pred)
                    self.batch_metrics[metric_key].append(result)
                    results[metric_key] = result
                    
                elif metric_name in self.logit_metrics:
                    # For perplexity, use the logits
                    logit_key = f"{task}_logits"
                    y_pred = model_outputs.get(logit_key, None)
                    if y_pred is not None:
                        result = metric(y_true[task], y_pred)
                        self.batch_metrics[metric_key].append(result)
                        results[metric_key] = result
                    
                elif metric_name in self.classification_metrics:
                    # For accuracy-like metrics
                    logit_key = f"{task}_logits"
                    if logit_key in model_outputs:
                        y_pred = model_outputs[logit_key]
                        result = metric(y_true[task], y_pred)
                    else:
                        # Fallback to exact token match if no logits
                        token_match = (model_outputs[task] == y_true[task]).float().mean()
                        result = token_match
                    
                    self.batch_metrics[metric_key].append(result)
                    results[metric_key] = result
                
                elif metric_name in self.regression_metrics:
                    # For regression metrics like MSE, MAE
                    if task in model_outputs:
                        result = metric(y_true[task], model_outputs[task])
                        self.batch_metrics[metric_key].append(result)
                        results[metric_key] = result
        
        return results

    def get_metric_string(self, epoch=None, decimal_places=2, batch_metrics=False):
        """
        Generate a formatted string of metrics
        
        Args:
            epoch: Optional epoch number to include
            decimal_places: Number of decimal places for values
            batch_metrics: Whether to use the latest batch metrics
            
        Returns:
            Formatted string of metrics
        """
        string = f"Epoch {epoch} - " if epoch else ""
        
        # Get all metric keys (task_metric format)
        metric_keys = list(self.batch_metrics.keys())
        
        if batch_metrics:
            # Print latest values of batch metrics
            for metric_key in metric_keys:
                if self.batch_metrics[metric_key]:  # Check if list is not empty
                    latest_value = self.batch_metrics[metric_key][-1]
                    # Convert to float if it's a tensor
                    if isinstance(latest_value, torch.Tensor):
                        latest_value = latest_value.float().item()
                    string += f"{metric_key}: {latest_value:.{decimal_places}f} - "
        else:
            # Print mean of epoch metrics
            for metric_key in metric_keys:
                if self.batch_metrics[metric_key]:  # Check if list is not empty
                    # Convert values to float tensor before calculating mean
                    values = [v.float() if isinstance(v, torch.Tensor) else float(v) 
                             for v in self.batch_metrics[metric_key]]
                    values_tensor = torch.tensor(values, dtype=torch.float32)
                    mean_value = values_tensor.mean().item()
                    string += f"{metric_key}: {mean_value:.{decimal_places}f} - "
        
        return string[:-3] if string.endswith(" - ") else string
    
    def add_metric(self, name, value):
        self.metrics[name] = value