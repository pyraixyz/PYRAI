from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .distillation_analyzer import DistillationAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import jinja2
import json
import time
import torch.cuda as cuda
import pynvml
import scipy.stats

class QualityMetric(Enum):
    """Knowledge distillation quality metrics."""
    KNOWLEDGE_RETENTION = "retention"  # How well student retains teacher knowledge
    COMPRESSION_QUALITY = "compression"  # Quality of model compression
    GENERALIZATION_GAP = "generalization"  # Generalization capability
    ROBUSTNESS = "robustness"  # Model robustness
    EFFICIENCY = "efficiency"  # Computational efficiency

@dataclass
class QualityConfig:
    """Quality analysis configuration."""
    metrics: List[QualityMetric]
    threshold_retention: float = 0.8
    threshold_compression: float = 0.5
    threshold_generalization: float = 0.1
    threshold_robustness: float = 0.2
    n_perturbations: int = 100
    perturbation_scale: float = 0.1

class QualityAnalyzer:
    """A comprehensive analyzer for evaluating and optimizing knowledge distillation.
    
    This class provides tools for analyzing various aspects of knowledge distillation:
    - Quality metrics evaluation
    - Performance analysis
    - Model compression analysis
    - Training dynamics analysis
    - Prediction behavior analysis
    - Automated optimization
    - Interactive visualization
    
    Attributes:
        config (QualityConfig): Configuration for quality analysis
        base_analyzer (DistillationAnalyzer): Base analyzer for distillation metrics
    
    Example:
        >>> analyzer = QualityAnalyzer(config)
        >>> results = analyzer.analyze_quality(student_model, teacher_model, test_loader, device)
        >>> analyzer.generate_interactive_report(results, "reports/")
    """
    
    def __init__(
        self,
        base_analyzer: DistillationAnalyzer,
        config: QualityConfig
    ):
        """Initialize quality analyzer.
        
        Args:
            base_analyzer: Base distillation analyzer
            config: Quality analysis configuration
        """
        self.base_analyzer = base_analyzer
        self.config = config
        
    def analyze_quality(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze the quality of knowledge distillation.
        
        Evaluates multiple aspects of distillation quality including:
        - Knowledge retention
        - Compression efficiency
        - Generalization capability
        - Model robustness
        
        Args:
            student_model: The student model being evaluated
            teacher_model: The teacher model used for distillation
            test_loader: DataLoader for test data
            device: Device to run analysis on
        
        Returns:
            Dict containing quality metrics and detailed analysis results
            
        Example:
            >>> quality = analyzer.analyze_quality(student, teacher, test_loader, device)
            >>> print(f"Retention score: {quality['retention']['score']}")
        """
        quality = {}
        
        # Analyze knowledge retention
        if QualityMetric.KNOWLEDGE_RETENTION in self.config.metrics:
            quality["retention"] = self._analyze_knowledge_retention(
                student_model, teacher_model,
                test_loader, device
            )
            
        # Analyze compression quality
        if QualityMetric.COMPRESSION_QUALITY in self.config.metrics:
            quality["compression"] = self._analyze_compression_quality(
                student_model, teacher_model,
                test_loader, device
            )
            
        # Analyze generalization gap
        if QualityMetric.GENERALIZATION_GAP in self.config.metrics:
            quality["generalization"] = self._analyze_generalization_gap(
                student_model, teacher_model,
                test_loader, device
            )
            
        # Analyze robustness
        if QualityMetric.ROBUSTNESS in self.config.metrics:
            quality["robustness"] = self._analyze_robustness(
                student_model, teacher_model,
                test_loader, device
            )
            
        # Analyze efficiency
        if QualityMetric.EFFICIENCY in self.config.metrics:
            quality["efficiency"] = self._analyze_efficiency(
                student_model, teacher_model
            )
            
        # Compute overall quality score
        quality["overall_score"] = self._compute_overall_score(quality)
        
        return quality
        
    def _analyze_knowledge_retention(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze how well the student retains teacher knowledge.
        
        Evaluates knowledge retention through multiple perspectives:
        - Feature-level similarity
        - Prediction alignment
        - Response pattern matching
        - Knowledge transfer quality
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model as knowledge source
            test_loader: Test data loader
            device: Computation device
            
        Returns:
            Dict containing:
                - feature_similarity: Layer-wise feature alignment scores
                - prediction_alignment: Prediction agreement metrics
                - response_patterns: Response pattern analysis
                - transfer_quality: Overall knowledge transfer score
        """
        retention = {}
        
        student_model.eval()
        teacher_model.eval()
        
        total_agreement = 0
        total_kl_div = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                
                # Get predictions
                student_logits = student_model(data)
                teacher_logits = teacher_model(data)
                
                # Compute prediction agreement
                student_preds = student_logits.argmax(dim=1)
                teacher_preds = teacher_logits.argmax(dim=1)
                agreement = (student_preds == teacher_preds).float().mean()
                
                # Compute KL divergence
                kl_div = F.kl_div(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits, dim=1),
                    reduction='batchmean'
                )
                
                batch_size = data.size(0)
                total_agreement += agreement * batch_size
                total_kl_div += kl_div * batch_size
                total_samples += batch_size
                
        retention["prediction_agreement"] = float(total_agreement / total_samples)
        retention["kl_divergence"] = float(total_kl_div / total_samples)
        retention["retention_score"] = float(
            retention["prediction_agreement"] * (1 - retention["kl_divergence"])
        )
        
        return retention
        
    def _analyze_compression_quality(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze quality of model compression.
        
        Evaluates compression from multiple aspects:
        - Parameter reduction efficiency
        - Computational complexity reduction
        - Memory footprint reduction
        - Performance retention
        
        Args:
            student_model: Compressed student model
            teacher_model: Original teacher model
            test_loader: Test data loader
            device: Computation device
            
        Returns:
            Dict containing:
                - parameter_efficiency: Parameter reduction metrics
                - computational_efficiency: Computation reduction metrics
                - memory_efficiency: Memory usage metrics
                - performance_retention: Performance comparison metrics
        """
        compression = {}
        
        # Get model sizes
        student_size = sum(p.numel() for p in student_model.parameters())
        teacher_size = sum(p.numel() for p in teacher_model.parameters())
        compression_ratio = teacher_size / student_size
        
        # Get model performance
        student_acc = self._evaluate_model(student_model, test_loader, device)
        teacher_acc = self._evaluate_model(teacher_model, test_loader, device)
        
        compression["size_ratio"] = float(compression_ratio)
        compression["performance_ratio"] = float(student_acc / teacher_acc)
        compression["compression_score"] = float(
            compression["performance_ratio"] / compression["size_ratio"]
        )
        
        return compression
        
    def _analyze_generalization_gap(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze model generalization capability.
        
        Evaluates generalization through:
        - Performance on unseen data
        - Out-of-distribution behavior
        - Cross-domain adaptation
        - Confidence calibration
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            test_loader: Test data loader
            device: Computation device
            
        Returns:
            Dict containing:
                - unseen_performance: Performance on unseen data
                - ood_behavior: Out-of-distribution analysis
                - adaptation: Cross-domain adaptation metrics
                - calibration: Confidence calibration scores
        """
        generalization = {}
        
        # Split test data into seen and unseen classes
        seen_loader, unseen_loader = self._split_test_data(test_loader)
        
        # Evaluate on seen classes
        student_seen = self._evaluate_model(student_model, seen_loader, device)
        teacher_seen = self._evaluate_model(teacher_model, seen_loader, device)
        
        # Evaluate on unseen classes
        student_unseen = self._evaluate_model(student_model, unseen_loader, device)
        teacher_unseen = self._evaluate_model(teacher_model, unseen_loader, device)
        
        generalization["seen_gap"] = float(teacher_seen - student_seen)
        generalization["unseen_gap"] = float(teacher_unseen - student_unseen)
        generalization["generalization_score"] = float(
            1 - (generalization["unseen_gap"] / generalization["seen_gap"])
        )
        
        return generalization
        
    def _analyze_robustness(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze model robustness to perturbations.
        
        Tests robustness against:
        - Input perturbations
        - Adversarial attacks
        - Noise injection
        - Distribution shifts
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            test_loader: Test data loader
            device: Computation device
            
        Returns:
            Dict containing:
                - perturbation_stability: Stability under perturbations
                - adversarial_robustness: Adversarial attack resistance
                - noise_tolerance: Noise tolerance metrics
                - distribution_robustness: Distribution shift handling
        """
        robustness = {}
        
        student_model.eval()
        teacher_model.eval()
        
        total_stability = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                batch_size = data.size(0)
                
                # Generate perturbations
                perturbations = []
                for _ in range(self.config.n_perturbations):
                    noise = torch.randn_like(data) * self.config.perturbation_scale
                    perturbations.append(data + noise)
                    
                # Get predictions
                student_preds = []
                teacher_preds = []
                
                for perturbed_data in perturbations:
                    student_logits = student_model(perturbed_data)
                    teacher_logits = teacher_model(perturbed_data)
                    
                    student_preds.append(student_logits.argmax(dim=1))
                    teacher_preds.append(teacher_logits.argmax(dim=1))
                    
                # Compute prediction stability
                student_stability = self._compute_stability(student_preds)
                teacher_stability = self._compute_stability(teacher_preds)
                
                relative_stability = student_stability / teacher_stability
                total_stability += relative_stability * batch_size
                total_samples += batch_size
                
        robustness["relative_stability"] = float(total_stability / total_samples)
        robustness["robustness_score"] = float(
            min(robustness["relative_stability"], 1.0)
        )
        
        return robustness
        
    def _analyze_efficiency(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze computational and memory efficiency.
        
        Evaluates efficiency in terms of:
        - Inference speed
        - Memory usage
        - Hardware utilization
        - Energy consumption
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            
        Returns:
            Dict containing:
                - inference: Inference speed metrics
                - memory: Memory usage analysis
                - hardware: Hardware utilization stats
                - energy: Energy consumption estimates
        """
        efficiency = {}
        
        # Calculate FLOPs
        student_flops = self.base_analyzer._calculate_flops(student_model)
        teacher_flops = self.base_analyzer._calculate_flops(teacher_model)
        
        # Calculate memory usage
        student_memory = sum(p.nelement() * p.element_size() for p in student_model.parameters())
        teacher_memory = sum(p.nelement() * p.element_size() for p in teacher_model.parameters())
        
        efficiency["flops_ratio"] = float(teacher_flops / student_flops)
        efficiency["memory_ratio"] = float(teacher_memory / student_memory)
        efficiency["efficiency_score"] = float(
            (efficiency["flops_ratio"] + efficiency["memory_ratio"]) / 2
        )
        
        return efficiency
        
    def _compute_overall_score(
        self,
        quality: Dict[str, Any]
    ) -> float:
        """Compute overall quality score from individual metrics.
        
        Combines multiple quality aspects:
        - Knowledge retention
        - Compression quality
        - Generalization capability
        - Model robustness
        - Computational efficiency
        
        Args:
            quality: Dictionary containing individual quality metrics
            
        Returns:
            Float between 0 and 1 representing overall quality score
            
        Note:
            Weights different aspects based on configuration thresholds
        """
        scores = []
        weights = {
            "retention": 0.3,
            "compression": 0.2,
            "generalization": 0.2,
            "robustness": 0.15,
            "efficiency": 0.15
        }
        
        for metric, result in quality.items():
            if metric == "overall_score":
                continue
                
            if f"{metric}_score" in result:
                scores.append(
                    result[f"{metric}_score"] * weights[metric]
                )
                
        return float(sum(scores))
        
    @staticmethod
    def _evaluate_model(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        return correct / total
        
    @staticmethod
    def _split_test_data(
        test_loader: torch.utils.data.DataLoader
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Split test data into seen and unseen classes."""
        # This is a placeholder - actual implementation would depend on the dataset
        return test_loader, test_loader
        
    @staticmethod
    def _compute_stability(
        predictions: List[torch.Tensor]
    ) -> float:
        """Compute prediction stability."""
        predictions = torch.stack(predictions)
        mode_preds, _ = torch.mode(predictions, dim=0)
        stability = torch.mean(
            torch.mean((predictions == mode_preds).float(), dim=0)
        ).item()
        return stability

    def _analyze_feature_quality(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze quality of learned features."""
        feature_quality = {}
        
        student_model.eval()
        teacher_model.eval()
        
        # Collect features from intermediate layers
        student_features = []
        teacher_features = []
        
        def hook_fn(module, input, output, features):
            features.append(output.detach())
        
        # Register hooks
        student_hooks = []
        teacher_hooks = []
        
        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(
                    lambda m, i, o, f=student_features: hook_fn(m, i, o, f)
                )
                student_hooks.append(hook)
                
        for name, module in teacher_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(
                    lambda m, i, o, f=teacher_features: hook_fn(m, i, o, f)
                )
                teacher_hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                student_model(data)
                teacher_model(data)
                break
        
        # Remove hooks
        for hook in student_hooks + teacher_hooks:
            hook.remove()
        
        # Analyze feature similarity
        feature_quality["layer_similarity"] = self._analyze_layer_similarity(
            student_features, teacher_features
        )
        
        # Analyze feature statistics
        feature_quality["statistics"] = self._analyze_feature_statistics(
            student_features, teacher_features
        )
        
        # Compute feature quality score
        feature_quality["feature_score"] = self._compute_feature_score(
            feature_quality["layer_similarity"],
            feature_quality["statistics"]
        )
        
        return feature_quality

    def _analyze_layer_similarity(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Analyze feature similarity between corresponding layers.
        
        Computes similarity using:
        - CKA (Centered Kernel Alignment)
        - Feature correlation
        - Statistical moments
        - Distribution matching
        
        Args:
            student_features: List of student model layer features
            teacher_features: List of teacher model layer features
            
        Returns:
            Dict containing:
                - cka: CKA similarity scores
                - correlation: Feature correlation metrics
                - moments: Statistical moment differences
                - distribution: Distribution similarity scores
        """
        similarities = {
            "cka": [],
            "correlation": [],
            "cosine": []
        }
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Reshape features
            s_flat = s_feat.reshape(s_feat.size(0), -1)
            t_flat = t_feat.reshape(t_feat.size(0), -1)
            
            # Compute CKA similarity
            cka = self.base_analyzer._compute_cka_similarity(
                s_flat.cpu().numpy(),
                t_flat.cpu().numpy()
            )
            similarities["cka"].append(float(cka))
            
            # Compute correlation
            correlation = torch.corrcoef(torch.stack([
                s_flat.mean(dim=0),
                t_flat.mean(dim=0)
            ]))[0, 1].item()
            similarities["correlation"].append(float(correlation))
            
            # Compute cosine similarity
            cosine = F.cosine_similarity(
                s_flat.mean(dim=0, keepdim=True),
                t_flat.mean(dim=0, keepdim=True)
            ).item()
            similarities["cosine"].append(float(cosine))
        
        return similarities

    def _analyze_feature_statistics(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Analyze feature statistics."""
        stats = {
            "student": {
                "mean": [],
                "std": [],
                "sparsity": [],
                "range": []
            },
            "teacher": {
                "mean": [],
                "std": [],
                "sparsity": [],
                "range": []
            }
        }
        
        for features, model in [
            (student_features, "student"),
            (teacher_features, "teacher")
        ]:
            for feat in features:
                feat_flat = feat.reshape(feat.size(0), -1)
                
                stats[model]["mean"].append(float(feat_flat.mean()))
                stats[model]["std"].append(float(feat_flat.std()))
                stats[model]["sparsity"].append(
                    float((feat_flat == 0).float().mean())
                )
                stats[model]["range"].append(float(
                    feat_flat.max() - feat_flat.min()
                ))
        
        return stats

    def _compute_feature_score(
        self,
        similarities: Dict[str, List[float]],
        statistics: Dict[str, Dict[str, List[float]]]
    ) -> float:
        """Compute overall feature quality score."""
        # Weight different similarity metrics
        similarity_weights = {
            "cka": 0.4,
            "correlation": 0.3,
            "cosine": 0.3
        }
        
        similarity_score = sum(
            np.mean(similarities[metric]) * weight
            for metric, weight in similarity_weights.items()
        )
        
        # Compare feature statistics
        stat_diffs = []
        for stat in ["mean", "std", "sparsity", "range"]:
            student_vals = np.array(statistics["student"][stat])
            teacher_vals = np.array(statistics["teacher"][stat])
            
            # Compute normalized difference
            diff = np.mean(np.abs(student_vals - teacher_vals) / (
                np.abs(teacher_vals) + 1e-10
            )
            stat_diffs.append(diff)
        
        statistics_score = 1 - np.mean(stat_diffs)
        
        # Combine scores
        return float(0.6 * similarity_score + 0.4 * statistics_score)

    def analyze_quality_over_time(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_steps: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze quality metrics over training steps."""
        quality_history = {
            metric.value: []
            for metric in self.config.metrics
        }
        
        for step in range(n_steps):
            # This is a placeholder - in practice, you would analyze
            # the model at different training checkpoints
            quality = self.analyze_quality(
                student_model,
                teacher_model,
                test_loader,
                device
            )
            
            for metric in self.config.metrics:
                quality_history[metric.value].append(
                    quality[metric.value]
                )
        
        return quality_history

    def plot_quality_metrics(
        self,
        quality: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Generate quality analysis plots."""
        plots = {}
        
        # Plot overall quality radar chart
        plots["overall"] = self._plot_quality_radar(quality)
        
        # Plot detailed metrics
        for metric in self.config.metrics:
            if metric.value in quality:
                plots[metric.value] = self._plot_metric_details(
                    metric.value,
                    quality[metric.value]
                )
        
        return plots

    def _plot_quality_radar(
        self,
        quality: Dict[str, Any]
    ) -> go.Figure:
        """Plot radar chart of quality metrics."""
        metrics = []
        scores = []
        
        for metric, result in quality.items():
            if metric != "overall_score" and f"{metric}_score" in result:
                metrics.append(metric)
                scores.append(result[f"{metric}_score"])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=metrics,
            fill='toself',
            name='Quality Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Overall Quality Analysis"
        )
        
        return fig

    def _plot_metric_details(
        self,
        metric: str,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot detailed analysis for specific metric."""
        if metric == "retention":
            return self._plot_retention_details(result)
        elif metric == "compression":
            return self._plot_compression_details(result)
        elif metric == "generalization":
            return self._plot_generalization_details(result)
        elif metric == "robustness":
            return self._plot_robustness_details(result)
        else:  # efficiency
            return self._plot_efficiency_details(result)

    def _plot_retention_details(
        self,
        result: Dict[str, float]
    ) -> go.Figure:
        """Plot knowledge retention details."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Prediction Agreement', 'KL Divergence')
        )
        
        # Plot prediction agreement
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=result["prediction_agreement"],
                title={'text': "Agreement"},
                gauge={'axis': {'range': [0, 1]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Plot KL divergence
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=result["kl_divergence"],
                title={'text': "KL Divergence"},
                gauge={'axis': {'range': [0, 2]}},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Knowledge Retention Analysis",
            height=400
        )
        
        return fig

    def _plot_compression_details(
        self,
        result: Dict[str, float]
    ) -> go.Figure:
        """Plot compression quality details."""
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["size_ratio"],
            title={'text': "Size Ratio"},
            gauge={'axis': {'range': [0, 2]}},
            domain={'row': 0, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["performance_ratio"],
            title={'text': "Performance Ratio"},
            gauge={'axis': {'range': [0, 2]}},
            domain={'row': 0, 'column': 1}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["compression_score"],
            title={'text': "Compression Score"},
            gauge={'axis': {'range': [0, 1]}},
            domain={'row': 0, 'column': 2}
        ))
        
        fig.update_layout(
            title="Compression Quality Analysis",
            height=400
        )
        
        return fig

    def _plot_generalization_details(
        self,
        result: Dict[str, float]
    ) -> go.Figure:
        """Plot generalization gap details."""
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["seen_gap"],
            title={'text': "Seen Gap"},
            gauge={'axis': {'range': [0, 2]}},
            domain={'row': 0, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["unseen_gap"],
            title={'text': "Unseen Gap"},
            gauge={'axis': {'range': [0, 2]}},
            domain={'row': 0, 'column': 1}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["generalization_score"],
            title={'text': "Generalization Score"},
            gauge={'axis': {'range': [0, 1]}},
            domain={'row': 0, 'column': 2}
        ))
        
        fig.update_layout(
            title="Generalization Gap Analysis",
            height=400
        )
        
        return fig

    def _plot_robustness_details(
        self,
        result: Dict[str, float]
    ) -> go.Figure:
        """Plot robustness details."""
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["relative_stability"],
            title={'text': "Relative Stability"},
            gauge={'axis': {'range': [0, 1]}},
            domain={'row': 0, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["robustness_score"],
            title={'text': "Robustness Score"},
            gauge={'axis': {'range': [0, 1]}},
            domain={'row': 0, 'column': 1}
        ))
        
        fig.update_layout(
            title="Robustness Analysis",
            height=400
        )
        
        return fig

    def _plot_efficiency_details(
        self,
        result: Dict[str, float]
    ) -> go.Figure:
        """Plot efficiency details."""
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["flops_ratio"],
            title={'text': "Flops Ratio"},
            gauge={'axis': {'range': [0, 2]}},
            domain={'row': 0, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["memory_ratio"],
            title={'text': "Memory Ratio"},
            gauge={'axis': {'range': [0, 2]}},
            domain={'row': 0, 'column': 1}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result["efficiency_score"],
            title={'text': "Efficiency Score"},
            gauge={'axis': {'range': [0, 1]}},
            domain={'row': 0, 'column': 2}
        ))
        
        fig.update_layout(
            title="Efficiency Analysis",
            height=400
        )
        
        return fig

    def _analyze_prediction_behavior(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze prediction behavior of student vs teacher."""
        behavior = {}
        
        # Collect predictions
        student_preds = []
        teacher_preds = []
        student_probs = []
        teacher_probs = []
        true_labels = []
        
        student_model.eval()
        teacher_model.eval()
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Get predictions
                s_out = student_model(data)
                t_out = teacher_model(data)
                
                # Store predictions and probabilities
                student_preds.append(s_out.argmax(dim=1))
                teacher_preds.append(t_out.argmax(dim=1))
                student_probs.append(F.softmax(s_out, dim=1))
                teacher_probs.append(F.softmax(t_out, dim=1))
                true_labels.append(labels)
        
        # Concatenate results
        student_preds = torch.cat(student_preds)
        teacher_preds = torch.cat(teacher_preds)
        student_probs = torch.cat(student_probs)
        teacher_probs = torch.cat(teacher_probs)
        true_labels = torch.cat(true_labels)
        
        # Analyze prediction diversity
        behavior["diversity"] = self._analyze_prediction_diversity(
            student_preds, teacher_preds,
            student_probs.max(dim=1)[0],
            teacher_probs.max(dim=1)[0]
        )
        
        # Analyze calibration
        behavior["calibration"] = self._analyze_prediction_calibration(
            student_probs, teacher_probs, true_labels
        )
        
        # Analyze error patterns
        behavior["error_analysis"] = self._analyze_error_patterns(
            student_preds, teacher_preds, true_labels
        )
        
        return behavior

    def _plot_prediction_behavior(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Generate prediction behavior plots."""
        plots = {}
        
        # Plot prediction diversity
        plots["diversity"] = self._plot_prediction_diversity(
            prediction["diversity"]
        )
        
        # Plot calibration
        plots["calibration"] = self._plot_calibration(
            prediction["calibration"]
        )
        
        # Plot error patterns
        plots["errors"] = self._plot_error_patterns(
            prediction["error_analysis"]
        )
        
        return plots

    def _plot_prediction_diversity(
        self,
        diversity: Dict[str, Any]
    ) -> go.Figure:
        """Plot prediction diversity analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Agreement Rate',
                'Confidence Distribution',
                'Disagreement Analysis',
                'Confidence Correlation'
            )
        )
        
        # Plot agreement rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=diversity["agreement_rate"] * 100,
                title={'text': "Agreement Rate (%)"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Plot confidence patterns
        patterns = diversity["confidence_patterns"]
        fig.add_trace(
            go.Bar(
                x=list(patterns.keys()),
                y=list(patterns.values()),
                name="Confidence Patterns"
            ),
            row=1, col=2
        )
        
        # Plot disagreement analysis
        if "disagreement_analysis" in diversity:
            disagree = diversity["disagreement_analysis"]
            fig.add_trace(
                go.Bar(
                    x=["Student", "Teacher"],
                    y=[
                        disagree["student_confidence"],
                        disagree["teacher_confidence"]
                    ],
                    name="Disagreement Confidence"
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=800,
            title="Prediction Diversity Analysis",
            showlegend=True
        )
        
        return fig

    def _analyze_prediction_calibration(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze prediction calibration between student and teacher models.
        
        Evaluates how well model confidence aligns with accuracy by:
        - Computing reliability diagrams
        - Calculating Expected Calibration Error (ECE)
        - Analyzing confidence distributions
        
        Args:
            student_probs: Student model probabilities (N x C)
            teacher_probs: Teacher model probabilities (N x C) 
            true_labels: Ground truth labels (N)
            
        Returns:
            Dict containing:
                - student: Student calibration statistics
                - teacher: Teacher calibration statistics
                Each containing:
                    - confidences: Mean confidence per bin
                    - accuracies: Mean accuracy per bin
                    - counts: Sample count per bin
                    - ece: Expected Calibration Error
        """
        def compute_calibration(probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
            confidences = probs.max(dim=1)[0].cpu().numpy()
            accuracies = (probs.argmax(dim=1) == labels).float().cpu().numpy()
            
            # Compute reliability diagram
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_stats = {
                "confidences": [],
                "accuracies": [],
                "counts": []
            }
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = np.logical_and(
                    confidences > bin_lower,
                    confidences <= bin_upper
                )
                if np.sum(in_bin) > 0:
                    calibration_stats["confidences"].append(
                        np.mean(confidences[in_bin])
                    )
                    calibration_stats["accuracies"].append(
                        np.mean(accuracies[in_bin])
                    )
                    calibration_stats["counts"].append(
                        np.sum(in_bin)
                    )
            
            # Compute calibration error
            calibration_stats["ece"] = np.abs(
                np.array(calibration_stats["confidences"]) -
                np.array(calibration_stats["accuracies"])
            ).mean()
            
            return calibration_stats
        
        return {
            "student": compute_calibration(student_probs, true_labels),
            "teacher": compute_calibration(teacher_probs, true_labels)
        }

    def _plot_calibration(
        self,
        calibration: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Plot calibration analysis results.
        
        Creates reliability diagrams showing:
        - Confidence vs accuracy curves
        - Perfect calibration reference line
        - ECE scores
        - Sample distributions
        
        Args:
            calibration: Calibration analysis results from _analyze_prediction_calibration()
            
        Returns:
            Plotly figure with reliability diagrams for student and teacher models
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Student Calibration', 'Teacher Calibration')
        )
        
        for i, model in enumerate(["student", "teacher"], 1):
            stats = calibration[model]
            
            # Plot reliability diagram
            fig.add_trace(
                go.Scatter(
                    x=stats["confidences"],
                    y=stats["accuracies"],
                    mode="lines+markers",
                    name=f"{model} calibration"
                ),
                row=1, col=i
            )
            
            # Plot perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash"),
                    name="Perfect calibration"
                ),
                row=1, col=i
            )
            
            # Add ECE annotation
            fig.add_annotation(
                text=f"ECE: {stats['ece']:.3f}",
                xref="x",
                yref="y",
                x=0.1,
                y=0.9,
                showarrow=False,
                row=1,
                col=i
            )
        
        fig.update_layout(
            height=400,
            title="Calibration Analysis",
            showlegend=True
        )
        
        return fig

    def _analyze_error_patterns(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze error patterns between student and teacher predictions.
        
        Examines:
        - Confusion matrices
        - Error type distributions
        - Class-wise error rates
        - Error correlations
        
        Args:
            student_preds: Student model predictions
            teacher_preds: Teacher model predictions
            true_labels: Ground truth labels
            
        Returns:
            Dict containing:
                - confusion: Confusion matrices for both models
                - error_types: Distribution of error types
                - class_errors: Per-class error analysis
        """
        patterns = {}
        
        # Compute confusion matrices
        student_confusion = self._compute_confusion_matrix(student_preds, true_labels)
        teacher_confusion = self._compute_confusion_matrix(teacher_preds, true_labels)
        
        patterns["confusion"] = {
            "student": student_confusion,
            "teacher": teacher_confusion
        }
        
        # Analyze error types
        student_errors = student_preds != true_labels
        teacher_errors = teacher_preds != true_labels
        
        patterns["error_types"] = {
            "student_only": float((student_errors & ~teacher_errors).float().mean()),
            "teacher_only": float((~student_errors & teacher_errors).float().mean()),
            "both": float((student_errors & teacher_errors).float().mean()),
            "neither": float((~student_errors & ~teacher_errors).float().mean())
        }
        
        # Analyze class-wise errors
        patterns["class_errors"] = self._analyze_class_errors(
            student_preds, teacher_preds, true_labels
        )
        
        return patterns

    def _analyze_class_errors(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """Analyze errors for each class."""
        class_errors = {}
        
        for class_idx in torch.unique(true_labels):
            class_mask = true_labels == class_idx
            
            class_errors[int(class_idx)] = {
                "student_error": float(
                    (student_preds[class_mask] != true_labels[class_mask]).float().mean()
                ),
                "teacher_error": float(
                    (teacher_preds[class_mask] != true_labels[class_mask]).float().mean()
                ),
                "agreement": float(
                    (student_preds[class_mask] == teacher_preds[class_mask]).float().mean()
                )
            }
        
        return class_errors

    def _plot_prediction_behavior(
        self,
        behavior: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Plot prediction behavior analysis."""
        plots = {}
        
        # Plot confusion matrices
        plots["confusion"] = self._plot_confusion_matrices(
            behavior["error_analysis"]["confusion"]
        )
        
        # Plot error types
        plots["error_types"] = self._plot_error_types(
            behavior["error_analysis"]["error_types"]
        )
        
        # Plot reliability diagrams
        plots["reliability"] = self._plot_reliability_diagrams(
            behavior["calibration"]["reliability"]
        )
        
        # Plot class errors
        plots["class_errors"] = self._plot_class_errors(
            behavior["error_analysis"]["class_errors"]
        )
        
        return plots

    def _plot_confusion_matrices(
        self,
        confusion: Dict[str, np.ndarray]
    ) -> go.Figure:
        """Plot confusion matrices."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Student Confusion Matrix', 'Teacher Confusion Matrix')
        )
        
        for i, (model, matrix) in enumerate(confusion.items(), 1):
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    colorscale="Viridis",
                    showscale=True,
                    name=model
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title="Confusion Matrices",
            height=500
        )
        
        return fig

    def _plot_reliability_diagrams(
        self,
        reliability: Dict[str, Dict[str, List[float]]]
    ) -> go.Figure:
        """Plot reliability diagrams."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Student Reliability', 'Teacher Reliability')
        )
        
        for i, (model, data) in enumerate(reliability.items(), 1):
            # Plot perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Perfect Calibration',
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
            
            # Plot actual calibration
            fig.add_trace(
                go.Scatter(
                    x=data["confidences"],
                    y=data["accuracies"],
                    mode='lines+markers',
                    name=f'{model} Calibration',
                    text=[f'n={c}' for c in data["counts"]],
                    hovertemplate='Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br>%{text}'
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title="Reliability Diagrams",
            height=400,
            xaxis_title="Confidence",
            xaxis2_title="Confidence",
            yaxis_title="Accuracy",
            yaxis2_title="Accuracy"
        )
        
        return fig

    def _plot_error_types(
        self,
        error_types: Dict[str, float]
    ) -> go.Figure:
        """Plot error types."""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(error_types.keys()),
            y=list(error_types.values()),
            name='Error Types'
        ))
        
        fig.update_layout(
            title="Error Types Analysis",
            xaxis_title="Error Type",
            yaxis_title="Percentage"
        )
        
        return fig

    def _plot_class_errors(
        self,
        class_errors: Dict[int, Dict[str, float]]
    ) -> go.Figure:
        """Plot class-wise errors."""
        fig = go.Figure()
        
        for class_idx, errors in class_errors.items():
            fig.add_trace(go.Bar(
                x=[f'Student Error', f'Teacher Error', f'Agreement'],
                y=[errors["student_error"], errors["teacher_error"], errors["agreement"]],
                name=f'Class {class_idx}'
            ))
        
        fig.update_layout(
            title="Class-wise Error Analysis",
            xaxis_title="Error Type",
            yaxis_title="Percentage"
        )
        
        return fig

    def _analyze_optimization_quality(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_history: List[Dict[str, float]],
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze quality of optimization process.
        
        Evaluates optimization effectiveness through:
        - Learning dynamics analysis
        - Gradient behavior analysis
        - Loss landscape analysis
        
        Args:
            student_model: Current student model
            teacher_model: Teacher model
            train_history: Training metrics history
            device: Computation device
            
        Returns:
            Dict containing:
                - learning: Learning dynamics analysis
                - gradients: Gradient behavior analysis
                - loss_landscape: Loss landscape analysis
                - optimization_score: Overall optimization quality score
        """
        optimization = {}
        
        # Analyze learning dynamics
        optimization["learning"] = self._analyze_learning_dynamics(train_history)
        
        # Analyze gradient behavior
        optimization["gradients"] = self._analyze_gradient_behavior(student_model, device)
        
        # Analyze loss landscape
        optimization["loss_landscape"] = self._analyze_loss_landscape(
            student_model, teacher_model, device
        )
        
        # Compute optimization score
        optimization["optimization_score"] = self._compute_optimization_score(
            optimization["learning"],
            optimization["gradients"],
            optimization["loss_landscape"]
        )
        
        return optimization

    def _analyze_gradient_behavior(
        self,
        model: nn.Module,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze gradient behavior during training.
        
        Examines gradient statistics and flow to identify potential
        optimization issues.
        
        Args:
            model: Model to analyze gradients
            device: Computation device
            
        Returns:
            Dict containing:
                - statistics: Gradient statistics
                - flow: Gradient flow analysis
                - vanishing: Vanishing gradient detection
        """
        gradients = {}
        
        # Collect gradient statistics
        grad_norms = []
        grad_means = []
        grad_vars = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_norms.append(grad.norm().item())
                grad_means.append(grad.mean().item())
                grad_vars.append(grad.var().item())
        
        gradients["statistics"] = {
            "norm": {
                "mean": float(np.mean(grad_norms)),
                "std": float(np.std(grad_norms)),
                "max": float(np.max(grad_norms)),
                "min": float(np.min(grad_norms))
            },
            "mean": {
                "mean": float(np.mean(grad_means)),
                "std": float(np.std(grad_means))
            },
            "variance": {
                "mean": float(np.mean(grad_vars)),
                "std": float(np.std(grad_vars))
            }
        }
        
        # Analyze gradient flow
        gradients["flow"] = self._analyze_gradient_flow(model)
        
        return gradients

    def _analyze_gradient_flow(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze gradient flow through network layers.
        
        Tracks how gradients propagate through the network to identify
        potential bottlenecks.
        
        Args:
            model: Model to analyze gradient flow
            
        Returns:
            Dict containing:
                - layer_stats: Layer-wise gradient statistics
                - propagation: Gradient propagation analysis
        """
        flow = {}
        
        # Collect layer-wise gradients
        layer_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer = name.split('.')[0]
                if layer not in layer_grads:
                    layer_grads[layer] = []
                layer_grads[layer].append(param.grad.detach())
        
        # Compute layer-wise statistics
        flow["layer_stats"] = {}
        for layer, grads in layer_grads.items():
            grad_tensor = torch.cat([g.flatten() for g in grads])
            flow["layer_stats"][layer] = {
                "norm": float(grad_tensor.norm().item()),
                "mean": float(grad_tensor.mean().item()),
                "std": float(grad_tensor.std().item()),
                "zero_fraction": float((grad_tensor == 0).float().mean().item())
            }
        
        # Analyze gradient propagation
        flow["propagation"] = self._analyze_gradient_propagation(layer_grads)
        
        return flow

    def _analyze_gradient_propagation(
        self,
        layer_grads: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, float]:
        """Analyze gradient propagation between layers.
        
        Computes correlation between gradients of adjacent layers to
        understand gradient propagation.
        
        Args:
            layer_grads: Dict mapping layer names to their gradients
        
        Returns:
            Dict mapping layer pairs to their gradient correlations
        """
        propagation = {}
        
        layers = list(layer_grads.keys())
        for i in range(len(layers)-1):
            curr_layer = layers[i]
            next_layer = layers[i+1]
            
            curr_grads = torch.cat([g.flatten() for g in layer_grads[curr_layer]])
            next_grads = torch.cat([g.flatten() for g in layer_grads[next_layer]])
            
            # Compute correlation between adjacent layers
            correlation = torch.corrcoef(torch.stack([
                curr_grads, next_grads
            ]))[0, 1].item()
            
            propagation[f"{curr_layer}_to_{next_layer}"] = correlation
        
        return propagation

    def _analyze_loss_landscape(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: torch.device,
        n_points: int = 20,
        alpha: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze loss landscape around current parameters."""
        landscape = {}
        
        # Get current parameters
        current_params = {
            name: param.clone()
            for name, param in student_model.named_parameters()
        }
        
        # Sample points around current parameters
        loss_values = []
        distances = []
        
        for _ in range(n_points):
            # Add random perturbation
            with torch.no_grad():
                for name, param in student_model.named_parameters():
                    noise = torch.randn_like(param) * alpha
                    param.copy_(current_params[name] + noise)
                    distances.append(noise.norm().item())
            
            # Compute loss at perturbed point
            loss = self._compute_distillation_loss(
                student_model, teacher_model, device
            )
            loss_values.append(loss)
            
            # Restore original parameters
            with torch.no_grad():
                for name, param in student_model.named_parameters():
                    param.copy_(current_params[name])
        
        # Analyze landscape characteristics
        landscape["statistics"] = {
            "mean_loss": float(np.mean(loss_values)),
            "std_loss": float(np.std(loss_values)),
            "min_loss": float(np.min(loss_values)),
            "max_loss": float(np.max(loss_values)),
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances))
        }
        
        # Analyze landscape smoothness
        landscape["smoothness"] = self._analyze_landscape_smoothness(
            loss_values, distances
        )
        
        return landscape

    def _analyze_landscape_smoothness(
        self,
        loss_values: List[float],
        distances: List[float]
    ) -> Dict[str, float]:
        """Analyze smoothness of loss landscape."""
        smoothness = {}
        
        # Compute loss-distance correlation
        correlation = np.corrcoef(loss_values, distances)[0, 1]
        smoothness["loss_distance_correlation"] = float(correlation)
        
        # Compute local Lipschitz constant
        lipschitz = np.max(np.abs(np.diff(loss_values)) / np.diff(distances))
        smoothness["local_lipschitz"] = float(lipschitz)
        
        # Compute variation ratio
        variation = np.std(loss_values) / np.mean(distances)
        smoothness["variation_ratio"] = float(variation)
        
        return smoothness

    def _compute_optimization_score(
        self,
        learning: Dict[str, Any],
        gradients: Dict[str, Any],
        landscape: Dict[str, Any]
    ) -> float:
        """Compute overall optimization quality score."""
        # Score learning dynamics
        learning_score = self._score_learning_dynamics(learning)
        
        # Score gradient behavior
        gradient_score = self._score_gradient_behavior(gradients)
        
        # Score loss landscape
        landscape_score = self._score_loss_landscape(landscape)
        
        # Combine scores with weights
        weights = {
            "learning": 0.4,
            "gradients": 0.3,
            "landscape": 0.3
        }
        
        return float(
            weights["learning"] * learning_score +
            weights["gradients"] * gradient_score +
            weights["landscape"] * landscape_score
        )

    def _score_learning_dynamics(
        self,
        learning: Dict[str, Any]
    ) -> float:
        """Score learning dynamics quality."""
        scores = []
        
        # Score convergence
        if "convergence" in learning:
            conv = learning["convergence"]
            scores.append(
                1.0 - min(conv["steps_to_converge"] / conv["total_steps"], 1.0)
            )
        
        # Score stability
        if "stability" in learning:
            stab = learning["stability"]
            scores.append(
                1.0 - min(stab["loss_std"] / stab["loss_mean"], 1.0)
            )
        
        # Score progress
        if "progress" in learning:
            prog = learning["progress"]
            scores.append(
                prog["improvement_ratio"]
            )
        
        return float(np.mean(scores)) if scores else 0.0

    def _score_gradient_behavior(
        self,
        gradients: Dict[str, Any]
    ) -> float:
        """Score gradient behavior quality."""
        scores = []
        
        # Score gradient statistics
        stats = gradients["statistics"]
        scores.append(
            1.0 - min(stats["norm"]["std"] / stats["norm"]["mean"], 1.0)
        )
        
        # Score gradient flow
        if "flow" in gradients:
            flow = gradients["flow"]
            layer_correlations = [
                v for k, v in flow["propagation"].items()
                if isinstance(v, float)
            ]
            scores.append(float(np.mean(layer_correlations)))
        
        return float(np.mean(scores)) if scores else 0.0

    def _score_loss_landscape(
        self,
        landscape: Dict[str, Any]
    ) -> float:
        """Score loss landscape quality."""
        scores = []
        
        # Score landscape statistics
        stats = landscape["statistics"]
        scores.append(
            1.0 - min(stats["std_loss"] / stats["mean_loss"], 1.0)
        )
        
        # Score landscape smoothness
        if "smoothness" in landscape:
            smooth = landscape["smoothness"]
            scores.append(
                1.0 - min(smooth["variation_ratio"], 1.0)
            )
            scores.append(
                1.0 - min(smooth["local_lipschitz"], 1.0)
            )
        
        return float(np.mean(scores)) if scores else 0.0

    def _compute_distillation_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: torch.device,
        batch_size: int = 32
    ) -> float:
        """Compute distillation loss at current parameters."""
        # Generate random input
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        
        student_model.eval()
        teacher_model.eval()
        
        with torch.no_grad():
            # Get model outputs
            student_logits = student_model(dummy_input)
            teacher_logits = teacher_model(dummy_input)
            
            # Compute KL divergence loss
            student_log_probs = F.log_softmax(student_logits, dim=1)
            teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=1)
            
            distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            )
            
        return float(distill_loss.item())

    def plot_optimization_analysis(
        self,
        optimization: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Generate optimization analysis plots."""
        plots = {}
        
        # Plot learning dynamics
        plots["learning"] = self._plot_learning_dynamics(
            optimization["learning"]
        )
        
        # Plot gradient analysis
        plots["gradients"] = self._plot_gradient_analysis(
            optimization["gradients"]
        )
        
        # Plot loss landscape
        plots["landscape"] = self._plot_loss_landscape(
            optimization["loss_landscape"]
        )
        
        # Plot overall optimization score
        plots["score"] = self._plot_optimization_score(optimization)
        
        return plots

    def _plot_learning_dynamics(
        self,
        learning: Dict[str, Any]
    ) -> go.Figure:
        """Plot learning dynamics analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss Convergence',
                'Loss Stability',
                'Training Progress',
                'Learning Rate'
            )
        )
        
        # Plot loss convergence
        if "convergence" in learning:
            conv = learning["convergence"]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(conv["loss_history"]))),
                    y=conv["loss_history"],
                    name="Loss",
                    mode="lines"
                ),
                row=1, col=1
            )
        
        # Plot loss stability
        if "stability" in learning:
            stab = learning["stability"]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(stab["loss_std_history"]))),
                    y=stab["loss_std_history"],
                    name="Loss Std",
                    mode="lines"
                ),
                row=1, col=2
            )
        
        # Plot training progress
        if "progress" in learning:
            prog = learning["progress"]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(prog["accuracy_history"]))),
                    y=prog["accuracy_history"],
                    name="Accuracy",
                    mode="lines"
                ),
                row=2, col=1
            )
        
        # Plot learning rate
        if "lr_history" in learning:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(learning["lr_history"]))),
                    y=learning["lr_history"],
                    name="Learning Rate",
                    mode="lines"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Learning Dynamics Analysis",
            showlegend=True
        )
        
        return fig

    def _plot_gradient_analysis(
        self,
        gradients: Dict[str, Any]
    ) -> go.Figure:
        """Plot gradient analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Gradient Norm',
                'Layer-wise Statistics',
                'Gradient Flow',
                'Zero Gradients'
            )
        )
        
        # Plot gradient norms
        stats = gradients["statistics"]["norm"]
        fig.add_trace(
            go.Bar(
                x=["mean", "std", "max", "min"],
                y=[stats["mean"], stats["std"], stats["max"], stats["min"]],
                name="Gradient Norms"
            ),
            row=1, col=1
        )
        
        # Plot layer-wise statistics
        flow = gradients["flow"]
        layers = list(flow["layer_stats"].keys())
        layer_norms = [flow["layer_stats"][l]["norm"] for l in layers]
        
        fig.add_trace(
            go.Bar(
                x=layers,
                y=layer_norms,
                name="Layer Norms"
            ),
            row=1, col=2
        )
        
        # Plot gradient flow
        correlations = []
        layer_pairs = []
        for k, v in flow["propagation"].items():
            if isinstance(v, float):
                correlations.append(v)
                layer_pairs.append(k)
        
        fig.add_trace(
            go.Heatmap(
                z=[correlations],
                x=layer_pairs,
                colorscale="RdBu",
                name="Gradient Flow"
            ),
            row=2, col=1
        )
        
        # Plot zero gradients
        zero_fracs = [flow["layer_stats"][l]["zero_fraction"] for l in layers]
        fig.add_trace(
            go.Bar(
                x=layers,
                y=zero_fracs,
                name="Zero Gradients"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Gradient Analysis",
            showlegend=True
        )
        
        return fig

    def _plot_loss_landscape(
        self,
        landscape: Dict[str, Any]
    ) -> go.Figure:
        """Plot loss landscape analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss Distribution',
                'Loss-Distance Correlation',
                'Landscape Smoothness',
                'Local Lipschitz'
            )
        )
        
        # Plot loss distribution
        stats = landscape["statistics"]
        fig.add_trace(
            go.Box(
                y=[stats["min_loss"], stats["mean_loss"], stats["max_loss"]],
                name="Loss Distribution"
            ),
            row=1, col=1
        )
        
        # Plot loss-distance correlation
        smooth = landscape["smoothness"]
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, smooth["loss_distance_correlation"]],
                name="Correlation",
                mode="lines"
            ),
            row=1, col=2
        )
        
        # Plot smoothness metrics
        fig.add_trace(
            go.Bar(
                x=["Variation Ratio", "Local Lipschitz"],
                y=[
                    smooth["variation_ratio"],
                    smooth["local_lipschitz"]
                ],
                name="Smoothness"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title="Loss Landscape Analysis",
            showlegend=True
        )
        
        return fig

    def _plot_optimization_score(
        self,
        optimization: Dict[str, Any]
    ) -> go.Figure:
        """Plot optimization quality score."""
        fig = go.Figure()
        
        scores = {
            "Learning": optimization["learning_score"],
            "Gradients": optimization["gradients_score"],
            "Landscape": optimization["landscape_score"],
            "Overall": optimization["optimization_score"]
        }
        
        fig.add_trace(
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                name="Quality Scores"
            )
        )
        
        fig.update_layout(
            title="Optimization Quality Scores",
            yaxis_title="Score",
            showlegend=False
        )
        
        return fig

    def analyze_and_recommend(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze distillation quality and provide optimization recommendations."""
        # Perform comprehensive analysis
        analysis = self.analyze_adaptively(
            student_model,
            teacher_model,
            test_loader,
            device
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        
        return {
            "analysis": analysis,
            "recommendations": recommendations
        }

    def _generate_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate optimization recommendations based on analysis results."""
        recommendations = {
            "critical": [],
            "important": [],
            "optional": []
        }
        
        # Analyze knowledge retention
        if "retention" in analysis:
            retention = analysis["retention"]
            if retention["retention_score"]["mean"] < self.config.threshold_retention:
                if retention["kl_divergence"]["mean"] > 1.0:
                    recommendations["critical"].append(
                        "High KL divergence detected. Consider:"
                        "\n- Increasing temperature parameter"
                        "\n- Adding intermediate layer distillation"
                        "\n- Using soft targets with higher weights"
                    )
                if retention["prediction_agreement"]["mean"] < 0.7:
                    recommendations["important"].append(
                        "Low prediction agreement. Consider:"
                        "\n- Adding attention transfer"
                        "\n- Using feature alignment"
                        "\n- Increasing distillation weight"
                    )
        
        # Analyze compression quality
        if "compression" in analysis:
            compression = analysis["compression"]
            if compression["compression_score"]["mean"] < self.config.threshold_compression:
                if compression["size_ratio"]["mean"] < 0.1:
                    recommendations["important"].append(
                        "Model might be too small. Consider:"
                        "\n- Increasing model capacity"
                        "\n- Using progressive compression"
                        "\n- Adding skip connections"
                    )
                if compression["performance_ratio"]["mean"] < 0.8:
                    recommendations["critical"].append(
                        "Significant performance drop. Consider:"
                        "\n- Using knowledge transfer"
                        "\n- Adding auxiliary tasks"
                        "\n- Fine-tuning on target task"
                    )
        
        # Analyze optimization quality
        if "optimization" in analysis:
            opt = analysis["optimization"]
            if "gradients" in opt:
                grad_stats = opt["gradients"]["statistics"]
                if grad_stats["norm"]["std"] / grad_stats["norm"]["mean"] > 0.5:
                    recommendations["important"].append(
                        "High gradient variance. Consider:"
                        "\n- Using gradient clipping"
                        "\n- Adjusting learning rate"
                        "\n- Adding batch normalization"
                    )
            if "loss_landscape" in opt:
                landscape = opt["loss_landscape"]
                if landscape["smoothness"]["variation_ratio"] > 0.3:
                    recommendations["important"].append(
                        "Rough loss landscape. Consider:"
                        "\n- Using learning rate warmup"
                        "\n- Adding regularization"
                        "\n- Implementing curriculum learning"
                    )
        
        # Add general recommendations
        if len(recommendations["critical"]) == 0 and len(recommendations["important"]) == 0:
            recommendations["optional"].extend([
                "Consider fine-tuning on specific tasks",
                "Experiment with different architectures",
                "Try ensemble distillation methods"
            ])
        
        return recommendations

    def generate_optimization_plan(
        self,
        recommendations: Dict[str, List[str]]
    ) -> str:
        """Generate structured optimization plan from recommendations."""
        template = """
        # Knowledge Distillation Optimization Plan

        ## Critical Issues
        {% if recommendations.critical %}
        {% for rec in recommendations.critical %}
        * {{ rec }}
        {% endfor %}
        {% else %}
        * No critical issues found
        {% endif %}

        ## Important Improvements
        {% if recommendations.important %}
        {% for rec in recommendations.important %}
        * {{ rec }}
        {% endfor %}
        {% else %}
        * No important improvements needed
        {% endif %}

        ## Optional Enhancements
        {% if recommendations.optional %}
        {% for rec in recommendations.optional %}
        * {{ rec }}
        {% endfor %}
        {% else %}
        * No optional enhancements suggested
        {% endif %}

        ## Implementation Timeline
        1. Address critical issues first (if any)
        2. Implement important improvements
        3. Consider optional enhancements based on resources

        ## Monitoring Metrics
        - Knowledge retention score
        - Compression efficiency
        - Inference performance
        - Model robustness
        """
        
        env = jinja2.Environment()
        template = env.from_string(template)
        return template.render(recommendations=recommendations)

    def export_analysis_summary(
        self,
        analysis: Dict[str, Any],
        recommendations: Dict[str, List[str]],
        output_path: Path
    ) -> None:
        """Export analysis summary and recommendations."""
        summary = {
            "quality_scores": {
                metric: scores["mean"]
                for metric, scores in analysis.items()
                if isinstance(scores, dict) and "mean" in scores
            },
            "critical_issues": len(recommendations["critical"]),
            "important_improvements": len(recommendations["important"]),
            "optional_enhancements": len(recommendations["optional"]),
            "recommendations": recommendations,
            "analysis_stats": analysis.get("analysis_stats", {})
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    def plot_optimization_recommendations(
        self,
        recommendations: Dict[str, List[str]]
    ) -> go.Figure:
        """Visualize optimization recommendations."""
        fig = go.Figure()
        
        # Create priority levels
        priority_levels = ["Critical", "Important", "Optional"]
        rec_counts = [
            len(recommendations["critical"]),
            len(recommendations["important"]),
            len(recommendations["optional"])
        ]
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=priority_levels,
            y=rec_counts,
            text=rec_counts,
            textposition='auto',
            marker_color=['red', 'orange', 'green']
        ))
        
        # Update layout
        fig.update_layout(
            title="Optimization Recommendations by Priority",
            xaxis_title="Priority Level",
            yaxis_title="Number of Recommendations",
            showlegend=False
        )
        
        return fig

    def analyze_adaptively(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        min_samples: int = 100,
        max_samples: int = 1000,
        target_confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Perform adaptive analysis with confidence-based sampling.
        
        Analyzes distillation quality with adaptive sampling until reaching
        desired confidence level or maximum samples.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            test_loader: Test data loader
            device: Computation device
            min_samples: Minimum number of samples to analyze
            max_samples: Maximum number of samples to analyze
            target_confidence: Target confidence level (0-1)
            
        Returns:
            Dict containing analysis results with confidence metrics
            
        Example:
            >>> results = analyzer.analyze_adaptively(
            ...     student, teacher, test_loader, device,
            ...     min_samples=50, target_confidence=0.9
            ... )
        """
        results = {}
        current_samples = 0
        confidence_scores = []
        
        while current_samples < max_samples:
            # Analyze with current batch
            batch_results = self.analyze_quality(
                student_model,
                teacher_model,
                test_loader,
                device
            )
            
            # Update results
            for metric, value in batch_results.items():
                if metric not in results:
                    results[metric] = []
                results[metric].append(value)
            
            # Compute confidence score
            confidence = self._compute_confidence_score(results)
            confidence_scores.append(confidence)
            
            # Check if we have enough confidence
            if current_samples >= min_samples and confidence >= target_confidence:
                break
            
            current_samples += 1
        
        # Aggregate results
        final_results = self._aggregate_results(results)
        final_results["analysis_stats"] = {
            "samples": current_samples,
            "confidence": confidence_scores[-1],
            "confidence_history": confidence_scores
        }
        
        return final_results

    def generate_analysis_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        format: str = "html"
    ) -> str:
        """Generate comprehensive analysis report."""
        # Create report directory
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plot_files = self._save_analysis_plots(results, report_dir)
        
        # Generate report content
        report_content = self._generate_report_content(
            results,
            plot_files
        )
        
        # Save report
        if format == "html":
            report_path = report_dir / "analysis_report.html"
            self._save_html_report(report_content, report_path)
        elif format == "pdf":
            report_path = report_dir / "analysis_report.pdf"
            self._save_pdf_report(report_content, report_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        return str(report_path)

    def _compute_confidence_score(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> float:
        """Compute confidence score for analysis results."""
        scores = []
        
        for metric, values in results.items():
            if len(values) < 2:
                continue
            
            # Compute variance of key metrics
            metric_scores = []
            for key in ["retention_score", "compression_score", 
                       "generalization_score", "robustness_score"]:
                if key in values[-1]:
                    variance = np.var([v[key] for v in values])
                    metric_scores.append(1.0 / (1.0 + variance))
            
            if metric_scores:
                scores.append(np.mean(metric_scores))
        
        return float(np.mean(scores)) if scores else 0.0

    def _aggregate_results(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Aggregate multiple analysis results."""
        aggregated = {}
        
        for metric, values in results.items():
            if not values:
                continue
            
            # Compute mean and std for numeric values
            metric_dict = {}
            for key, value in values[-1].items():
                if isinstance(value, (int, float)):
                    values_list = [v[key] for v in values]
                    metric_dict[key] = {
                        "mean": float(np.mean(values_list)),
                        "std": float(np.std(values_list)),
                        "min": float(np.min(values_list)),
                        "max": float(np.max(values_list))
                    }
                else:
                    metric_dict[key] = value
            
            aggregated[metric] = metric_dict
        
        return aggregated

    def _save_analysis_plots(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, str]:
        """Save analysis plots to files."""
        plot_files = {}
        
        # Generate and save all plots
        plots = {
            "quality": self.plot_quality_metrics(results),
            "optimization": self.plot_optimization_analysis(results),
            "prediction": self._plot_prediction_behavior(results.get("prediction", {}))
        }
        
        for plot_type, plot_dict in plots.items():
            plot_dir = output_dir / plot_type
            plot_dir.mkdir(exist_ok=True)
            
            for name, fig in plot_dict.items():
                file_path = plot_dir / f"{name}.html"
                fig.write_html(str(file_path))
                plot_files[f"{plot_type}_{name}"] = str(file_path)
        
        return plot_files

    def _generate_report_content(
        self,
        results: Dict[str, Any],
        plot_files: Dict[str, str]
    ) -> str:
        """Generate HTML report content."""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Knowledge Distillation Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                .plot { margin: 10px 0; }
                .metric { margin: 5px 0; }
                .score { font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Knowledge Distillation Analysis Report</h1>
            
            <div class="section">
                <h2>Overall Quality Scores</h2>
                {% for metric, score in quality_scores.items() %}
                <div class="metric">
                    <span class="name">{{ metric }}:</span>
                    <span class="score">{{ "%.3f"|format(score) }}</span>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Quality Analysis Plots</h2>
                {% for name, path in quality_plots.items() %}
                <div class="plot">
                    <iframe src="{{ path }}" width="100%" height="600px"></iframe>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Optimization Analysis</h2>
                {% for name, path in optimization_plots.items() %}
                <div class="plot">
                    <iframe src="{{ path }}" width="100%" height="600px"></iframe>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Prediction Behavior Analysis</h2>
                {% for name, path in prediction_plots.items() %}
                <div class="plot">
                    <iframe src="{{ path }}" width="100%" height="600px"></iframe>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Analysis Statistics</h2>
                <pre>{{ analysis_stats | tojson(indent=2) }}</pre>
            </div>
        </body>
        </html>
        """
        
        # Prepare template variables
        template_vars = {
            "quality_scores": {
                metric: result[f"{metric}_score"]["mean"]
                for metric, result in results.items()
                if f"{metric}_score" in result
            },
            "quality_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("quality_")
            },
            "optimization_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("optimization_")
            },
            "prediction_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("prediction_")
            },
            "analysis_stats": results.get("analysis_stats", {})
        }
        
        # Render template
        env = jinja2.Environment()
        template = env.from_string(template)
        return template.render(**template_vars)

    def _save_html_report(
        self,
        content: str,
        output_path: Path
    ) -> None:
        """Save HTML report to file."""
        with open(output_path, "w") as f:
            f.write(content)

    def _save_pdf_report(
        self,
        content: str,
        output_path: Path
    ) -> None:
        """Save PDF report to file."""
        try:
            import pdfkit
            pdfkit.from_string(content, str(output_path))
        except ImportError:
            print("Warning: pdfkit not installed. Saving as HTML instead.")
            self._save_html_report(content, output_path.with_suffix(".html"))

    def auto_optimize(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """Automatically optimize distillation hyperparameters."""
        from optuna import create_study, Trial
        
        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            config = self._sample_hyperparameters(trial)
            
            # Train student model
            student = self._train_student(
                student_model,
                teacher_model,
                train_loader,
                config,
                device
            )
            
            # Evaluate quality
            quality = self.analyze_quality(
                student,
                teacher_model,
                val_loader,
                device
            )
            
            return quality["overall_score"]
        
        # Create and run study
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "optimization_history": study.trials_dataframe().to_dict()
        }

    def _sample_hyperparameters(
        self,
        trial: Trial
    ) -> Dict[str, Any]:
        """Sample hyperparameters for optimization."""
        return {
            "temperature": trial.suggest_float("temperature", 1.0, 10.0),
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            "hard_weight": trial.suggest_float("hard_weight", 0.0, 1.0),
            "soft_weight": trial.suggest_float("soft_weight", 0.0, 1.0),
            "feature_weight": trial.suggest_float("feature_weight", 0.0, 1.0),
            "attention_weight": trial.suggest_float("attention_weight", 0.0, 1.0)
        }

    def debug_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Debug knowledge distillation issues.
        
        Performs comprehensive debugging analysis:
        - Architecture compatibility
        - Feature alignment
        - Gradient flow
        - Loss behavior
        
        Args:
            student_model: Student model to debug
            teacher_model: Teacher model for comparison
            data_loader: Data loader for analysis
            device: Computation device
            
        Returns:
            Dict containing debug information and recommendations
            
        Example:
            >>> debug_info = analyzer.debug_distillation(
            ...     student, teacher, train_loader, device
            ... )
            >>> print(debug_info["report"])
        """
        debug_info = {}
        
        # Check model architectures
        debug_info["architecture"] = self._check_architectures(
            student_model, teacher_model
        )
        
        # Check feature alignment
        debug_info["features"] = self._check_feature_alignment(
            student_model, teacher_model, data_loader, device
        )
        
        # Check gradient flow
        debug_info["gradients"] = self._check_gradient_flow(
            student_model, teacher_model, data_loader, device
        )
        
        # Check loss behavior
        debug_info["loss"] = self._check_loss_behavior(
            student_model, teacher_model, data_loader, device
        )
        
        # Generate debug report
        debug_info["report"] = self._generate_debug_report(debug_info)
        
        return debug_info

    def _check_architectures(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Check model architectures for potential issues."""
        issues = []
        warnings = []
        
        # Compare model sizes
        student_params = sum(p.numel() for p in student_model.parameters())
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        
        if student_params >= teacher_params:
            issues.append("Student model is larger than teacher")
        elif student_params < teacher_params * 0.1:
            warnings.append("Student model might be too small")
        
        # Check layer correspondence
        student_layers = [
            (name, module) 
            for name, module in student_model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        teacher_layers = [
            (name, module)
            for name, module in teacher_model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        
        if len(student_layers) > len(teacher_layers):
            issues.append("Student has more layers than teacher")
        
        # Check activation functions
        student_activations = set(
            type(m).__name__
            for m in student_model.modules()
            if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU))
        )
        teacher_activations = set(
            type(m).__name__
            for m in teacher_model.modules()
            if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU))
        )
        
        if student_activations != teacher_activations:
            warnings.append("Different activation functions used")
        
        return {
            "issues": issues,
            "warnings": warnings,
            "student_params": student_params,
            "teacher_params": teacher_params,
            "student_layers": len(student_layers),
            "teacher_layers": len(teacher_layers)
        }

    def _check_feature_alignment(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Check feature alignment between student and teacher."""
        alignment = {}
        
        # Collect intermediate features
        student_features = []
        teacher_features = []
        
        def hook_fn(features_list):
            def hook(module, input, output):
                features_list.append(output.detach())
        return hook
        
        # Register hooks
        student_hooks = []
        teacher_hooks = []
        
        for module in student_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn(student_features))
                student_hooks.append(hook)
        
        for module in teacher_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn(teacher_features))
                teacher_hooks.append(hook)
        
        # Forward pass
        try:
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(device)
                    student_model(data)
                    teacher_model(data)
                    break
        finally:
            # Remove hooks
            for hook in student_hooks + teacher_hooks:
                hook.remove()
        
        # Analyze feature statistics
        alignment["statistics"] = self._analyze_feature_statistics(
            student_features, teacher_features
        )
        
        # Check dimensionality
        alignment["dimensions"] = self._check_feature_dimensions(
            student_features, teacher_features
        )
        
        # Analyze feature correlations
        alignment["correlations"] = self._compute_feature_correlations(
            student_features, teacher_features
        )
        
        return alignment

    def _check_gradient_flow(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Check gradient flow issues."""
        flow = {}
        
        # Enable gradient computation
        student_model.train()
        teacher_model.eval()
        
        # Forward and backward pass
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Student forward pass
            student_outputs = student_model(data)
            teacher_outputs = teacher_model(data)
            
            # Compute loss
            loss = self._compute_distillation_loss(
                student_outputs, teacher_outputs, labels
            )
            
            # Backward pass
            loss.backward()
            break
        
        # Analyze gradients
        flow["gradient_norms"] = self._analyze_gradient_norms(student_model)
        flow["gradient_flow"] = self._analyze_gradient_propagation(student_model)
        flow["vanishing_gradients"] = self._check_vanishing_gradients(student_model)
        
        return flow

    def _check_loss_behavior(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Check loss behavior during training."""
        behavior = {}
        
        # Track loss components
        losses = []
        soft_losses = []
        hard_losses = []
        
        student_model.train()
        teacher_model.eval()
        
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            student_outputs = student_model(data)
            teacher_outputs = teacher_model(data)
            
            # Compute losses
            loss_info = self._compute_detailed_losses(
                student_outputs,
                teacher_outputs,
                labels
            )
            
            losses.append(loss_info["total"].item())
            soft_losses.append(loss_info["soft"].item())
            hard_losses.append(loss_info["hard"].item())
            
            if len(losses) >= 100:  # Collect statistics for 100 batches
                break
        
        # Analyze loss behavior
        behavior["loss_statistics"] = {
            "total": {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
                "min": float(np.min(losses)),
                "max": float(np.max(losses))
            },
            "soft": {
                "mean": float(np.mean(soft_losses)),
                "std": float(np.std(soft_losses))
            },
            "hard": {
                "mean": float(np.mean(hard_losses)),
                "std": float(np.std(hard_losses))
            }
        }
        
        # Check loss ratios
        behavior["loss_ratios"] = {
            "soft_hard_ratio": float(np.mean(soft_losses) / np.mean(hard_losses))
        }
        
        return behavior

    def _generate_debug_report(
        self,
        debug_info: Dict[str, Any]
    ) -> str:
        """Generate detailed debug report."""
        template = """
        # Knowledge Distillation Debug Report

        ## Architecture Issues
        {% if debug_info.architecture.issues %}
        Critical Issues:
        {% for issue in debug_info.architecture.issues %}
        * {{ issue }}
        {% endfor %}
        {% else %}
        No critical architecture issues found.
        {% endif %}

        {% if debug_info.architecture.warnings %}
        Warnings:
        {% for warning in debug_info.architecture.warnings %}
        * {{ warning }}
        {% endfor %}
        {% endif %}

        ## Feature Alignment
        - Feature dimension mismatches: {{ debug_info.features.dimensions.mismatches }}
        - Average feature correlation: {{ "%.3f"|format(debug_info.features.correlations.mean) }}
        
        ## Gradient Flow
        - Vanishing gradients detected: {{ debug_info.gradients.vanishing_gradients.detected }}
        - Average gradient norm: {{ "%.3f"|format(debug_info.gradients.gradient_norms.mean) }}
        
        ## Loss Behavior
        - Average total loss: {{ "%.3f"|format(debug_info.loss.loss_statistics.total.mean) }}
        - Soft/Hard loss ratio: {{ "%.3f"|format(debug_info.loss.loss_ratios.soft_hard_ratio) }}

        ## Recommendations
        {% for rec in recommendations %}
        * {{ rec }}
        {% endfor %}
        """
        
        # Generate recommendations
        recommendations = self._generate_debug_recommendations(debug_info)
        
        # Render template
        env = jinja2.Environment()
        template = env.from_string(template)
        return template.render(
            debug_info=debug_info,
            recommendations=recommendations
        )

    def _generate_debug_recommendations(
        self,
        debug_info: Dict[str, Any]
    ) -> List[str]:
        """Generate debugging recommendations."""
        recommendations = []
        
        # Architecture recommendations
        if debug_info["architecture"]["issues"]:
            recommendations.extend([
                f"Fix architecture issue: {issue}"
                for issue in debug_info["architecture"]["issues"]
            ])
        
        # Feature alignment recommendations
        if debug_info["features"]["dimensions"]["mismatches"] > 0:
            recommendations.append(
                "Add feature adaptation layers to match dimensions"
            )
        
        if debug_info["features"]["correlations"]["mean"] < 0.5:
            recommendations.append(
                "Improve feature alignment using intermediate supervision"
            )
        
        # Gradient flow recommendations
        if debug_info["gradients"]["vanishing_gradients"]["detected"]:
            recommendations.extend([
                "Add residual connections",
                "Adjust learning rate",
                "Consider gradient clipping"
            ])
        
        # Loss behavior recommendations
        loss_ratio = debug_info["loss"]["loss_ratios"]["soft_hard_ratio"]
        if loss_ratio < 0.1:
            recommendations.append("Increase soft target weight")
        elif loss_ratio > 10:
            recommendations.append("Increase hard target weight")
        
        return recommendations 

    # Add these methods to QualityAnalyzer class

    def _check_feature_dimensions(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Check feature dimension compatibility."""
        dimensions = {
            "mismatches": 0,
            "details": []
        }
        
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            s_shape = s_feat.shape
            t_shape = t_feat.shape
            
            if s_shape != t_shape:
                dimensions["mismatches"] += 1
                dimensions["details"].append({
                    "layer": i,
                    "student_shape": list(s_shape),
                    "teacher_shape": list(t_shape)
                })
        
        return dimensions

    def _compute_feature_correlations(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Compute correlations between student and teacher features.
        
        Calculates various correlation metrics between corresponding feature maps
        to measure feature alignment quality.
        
        Args:
            student_features: List of student model feature tensors
            teacher_features: List of teacher model feature tensors
            
        Returns:
            Dict containing:
                - pearson: Pearson correlation coefficients
                - spearman: Spearman rank correlations
                - cka: Centered Kernel Alignment scores
                - mean: Mean correlation across layers
                - std: Standard deviation of correlations
        """
        correlations = {
            "pearson": [],
            "spearman": [],
            "cka": [],
            "mean": 0.0,
            "std": 0.0
        }
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Reshape features
            s_flat = s_feat.reshape(s_feat.size(0), -1)
            t_flat = t_feat.reshape(t_feat.size(0), -1)
            
            # Compute Pearson correlation
            pearson = torch.corrcoef(torch.stack([
                s_flat.mean(dim=0),
                t_flat.mean(dim=0)
            ]))[0, 1].item()
            correlations["pearson"].append(pearson)
            
            # Compute Spearman correlation
            spearman = scipy.stats.spearmanr(s_flat.flatten(), t_flat.flatten())[0]
            correlations["spearman"].append(spearman)
            
            # Compute CKA similarity
            cka = self.base_analyzer._compute_cka_similarity(
                s_flat.cpu().numpy(),
                t_flat.cpu().numpy()
            )
            correlations["cka"].append(float(cka))
        
        if correlations["pearson"] and correlations["spearman"] and correlations["cka"]:
            correlations["mean"] = float(np.mean([correlations["pearson"], correlations["spearman"], correlations["cka"]]))
            correlations["std"] = float(np.std([correlations["pearson"], correlations["spearman"], correlations["cka"]]))
        
        return correlations

    def _analyze_gradient_norms(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze gradient norms across layers."""
        norms = {
            "layer_wise": {},
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": float('inf')
        }
        
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms.append(norm)
                norms["layer_wise"][name] = norm
                norms["max"] = max(norms["max"], norm)
                norms["min"] = min(norms["min"], norm)
        
        if grad_norms:
            norms["mean"] = float(np.mean(grad_norms))
            norms["std"] = float(np.std(grad_norms))
        
        return norms

    def _check_vanishing_gradients(
        self,
        model: nn.Module,
        threshold: float = 1e-4
    ) -> Dict[str, Any]:
        """Check for vanishing gradients."""
        vanishing = {
            "detected": False,
            "layers": [],
            "statistics": {
                "total_params": 0,
                "vanishing_params": 0,
                "ratio": 0.0
            }
        }
        
        total_params = 0
        vanishing_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_params += param.numel()
                small_grads = (param.grad.abs() < threshold).sum().item()
                vanishing_params += small_grads
                
                if small_grads / param.numel() > 0.5:  # More than 50% small gradients
                    vanishing["layers"].append({
                        "name": name,
                        "ratio": small_grads / param.numel()
                    })
        
        vanishing["detected"] = len(vanishing["layers"]) > 0
        vanishing["statistics"]["total_params"] = total_params
        vanishing["statistics"]["vanishing_params"] = vanishing_params
        vanishing["statistics"]["ratio"] = vanishing_params / total_params
        
        return vanishing

    def _compute_detailed_losses(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """Compute detailed loss components."""
        # Compute soft targets loss
        student_logits = student_outputs / temperature
        teacher_logits = teacher_outputs / temperature
        
        soft_loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Compute hard targets loss
        hard_loss = F.cross_entropy(student_outputs, labels)
        
        # Compute total loss
        total_loss = self.config.soft_weight * soft_loss + \
                     self.config.hard_weight * hard_loss
        
        return {
            "total": total_loss,
            "soft": soft_loss,
            "hard": hard_loss
        }

    def _train_student(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        n_epochs: int = 10
    ) -> nn.Module:
        """Train student model with given configuration."""
        optimizer = torch.optim.Adam(student_model.parameters())
        student_model.train()
        teacher_model.eval()
        
        for epoch in range(n_epochs):
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                student_outputs = student_model(data)
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                # Compute loss
                loss_info = self._compute_detailed_losses(
                    student_outputs,
                    teacher_outputs,
                    labels,
                    temperature=config["temperature"]
                )
                
                # Backward pass
                loss_info["total"].backward()
                optimizer.step()
        
        return student_model

    def _analyze_prediction_diversity(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        student_confs: torch.Tensor,
        teacher_confs: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze prediction diversity between student and teacher models.
        
        Examines prediction agreement and confidence patterns to understand how
        student predictions differ from teacher.
        
        Args:
            student_preds: Student model class predictions (N)
            teacher_preds: Teacher model class predictions (N)
            student_confs: Student model confidence scores (N)
            teacher_confs: Teacher model confidence scores (N)
        
        Returns:
            Dict containing:
                - agreement_rate: Overall prediction agreement ratio
                - confidence_patterns: Analysis of confidence distributions
                - disagreement_analysis: Analysis of disagreement cases
        """
        diversity = {}
        
        # Compute prediction agreement
        agreement = (student_preds == teacher_preds).float()
        diversity["agreement_rate"] = float(agreement.mean())
        
        # Analyze confidence patterns
        diversity["confidence_patterns"] = {
            "both_high": float((
                (student_confs > 0.9) &
                (teacher_confs > 0.9)
            ).float().mean()),
            "both_low": float((
                (student_confs < 0.5) &
                (teacher_confs < 0.5)
            ).float().mean()),
            "student_higher": float((
                student_confs > teacher_confs
            ).float().mean())
        }
        
        # Analyze disagreement patterns
        disagreement_mask = ~agreement.bool()
        if disagreement_mask.any():
            diversity["disagreement_analysis"] = {
                "student_confidence": float(
                    student_confs[disagreement_mask].mean()
                ),
                "teacher_confidence": float(
                    teacher_confs[disagreement_mask].mean()
                )
            }
        
        return diversity

    def analyze_performance(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        batch_sizes: List[int] = [1, 8, 16, 32, 64]
    ) -> Dict[str, Any]:
        """Analyze model performance characteristics."""
        performance = {}
        
        # Measure inference speed
        performance["inference"] = self._analyze_inference_speed(
            student_model, teacher_model, test_loader, device, batch_sizes
        )
        
        # Measure memory usage
        performance["memory"] = self._analyze_memory_usage(
            student_model, teacher_model, test_loader, device, batch_sizes
        )
        
        # Measure hardware utilization
        performance["utilization"] = self._analyze_hardware_utilization(
            student_model, teacher_model, device
        )
        
        return performance

    def _analyze_inference_speed(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        batch_sizes: List[int]
    ) -> Dict[str, Any]:
        """Analyze inference speed for different batch sizes."""
        import time
        
        speed = {
            "student": {"latency": {}, "throughput": {}},
            "teacher": {"latency": {}, "throughput": {}}
        }
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    student_model(dummy_input)
                    teacher_model(dummy_input)
            
            # Measure student model
            student_times = []
            for _ in range(100):
                start_time = time.perf_counter()
                with torch.no_grad():
                    student_model(dummy_input)
                student_times.append(time.perf_counter() - start_time)
            
            # Measure teacher model
            teacher_times = []
            for _ in range(100):
                start_time = time.perf_counter()
                with torch.no_grad():
                    teacher_model(dummy_input)
                teacher_times.append(time.perf_counter() - start_time)
            
            # Compute statistics
            student_latency = np.mean(student_times)
            teacher_latency = np.mean(teacher_times)
            
            speed["student"]["latency"][batch_size] = student_latency
            speed["student"]["throughput"][batch_size] = batch_size / student_latency
            speed["teacher"]["latency"][batch_size] = teacher_latency
            speed["teacher"]["throughput"][batch_size] = batch_size / teacher_latency
        
        return speed

    def _analyze_memory_usage(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        batch_sizes: List[int]
    ) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        import torch.cuda as cuda
        
        memory = {
            "student": {"peak": {}, "allocated": {}},
            "teacher": {"peak": {}, "allocated": {}}
        }
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Reset memory stats
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Measure student model
            with torch.no_grad():
                student_model(dummy_input)
                if device.type == "cuda":
                    memory["student"]["peak"][batch_size] = cuda.max_memory_allocated()
                    memory["student"]["allocated"][batch_size] = cuda.memory_allocated()
            
            # Reset memory stats
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Measure teacher model
            with torch.no_grad():
                teacher_model(dummy_input)
                if device.type == "cuda":
                    memory["teacher"]["peak"][batch_size] = cuda.max_memory_allocated()
                    memory["teacher"]["allocated"][batch_size] = cuda.memory_allocated()
        
        return memory

    def _analyze_hardware_utilization(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze hardware utilization patterns."""
        utilization = {
            "student": {},
            "teacher": {}
        }
        
        if device.type == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
                
                # Measure student model
                student_model(torch.randn(1, 3, 224, 224).to(device))
                student_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization["student"] = {
                    "gpu": student_util.gpu,
                    "memory": student_util.memory
                }
                
                # Measure teacher model
                teacher_model(torch.randn(1, 3, 224, 224).to(device))
                teacher_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization["teacher"] = {
                    "gpu": teacher_util.gpu,
                    "memory": teacher_util.memory
                }
                
                pynvml.nvmlShutdown()
            except ImportError:
                print("pynvml not installed. GPU utilization measurement skipped.")
        
        return utilization

    def plot_performance_analysis(
        self,
        performance: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Plot performance analysis results."""
        plots = {}
        
        # Plot inference speed
        plots["inference"] = self._plot_inference_speed(
            performance["inference"]
        )
        
        # Plot memory usage
        plots["memory"] = self._plot_memory_usage(
            performance["memory"]
        )
        
        # Plot hardware utilization
        plots["utilization"] = self._plot_hardware_utilization(
            performance["utilization"]
        )
        
        return plots

    def _plot_inference_speed(
        self,
        speed: Dict[str, Dict[str, Dict[int, float]]]
    ) -> go.Figure:
        """Plot inference speed analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Latency', 'Throughput')
        )
        
        # Plot latency
        for model in ["student", "teacher"]:
            batch_sizes = list(speed[model]["latency"].keys())
            latencies = list(speed[model]["latency"].values())
            
            fig.add_trace(
                go.Scatter(
                    x=batch_sizes,
                    y=latencies,
                    name=f"{model} latency",
                    mode="lines+markers"
                ),
                row=1, col=1
            )
        
        # Plot throughput
        for model in ["student", "teacher"]:
            batch_sizes = list(speed[model]["throughput"].keys())
            throughputs = list(speed[model]["throughput"].values())
            
            fig.add_trace(
                go.Scatter(
                    x=batch_sizes,
                    y=throughputs,
                    name=f"{model} throughput",
                    mode="lines+markers"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            title="Inference Speed Analysis",
            xaxis_title="Batch Size",
            xaxis2_title="Batch Size",
            yaxis_title="Latency (s)",
            yaxis2_title="Throughput (samples/s)"
        )
        
        return fig

    def _plot_memory_usage(
        self,
        memory: Dict[str, Dict[str, Dict[int, int]]]
    ) -> go.Figure:
        """Plot memory usage analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Peak Memory', 'Allocated Memory')
        )
        
        # Plot peak memory
        for model in ["student", "teacher"]:
            batch_sizes = list(memory[model]["peak"].keys())
            peak_memory = [m / 1024**2 for m in memory[model]["peak"].values()]  # Convert to MB
            
            fig.add_trace(
                go.Scatter(
                    x=batch_sizes,
                    y=peak_memory,
                    name=f"{model} peak",
                    mode="lines+markers"
                ),
                row=1, col=1
            )
        
        # Plot allocated memory
        for model in ["student", "teacher"]:
            batch_sizes = list(memory[model]["allocated"].keys())
            allocated_memory = [m / 1024**2 for m in memory[model]["allocated"].values()]
            
            fig.add_trace(
                go.Scatter(
                    x=batch_sizes,
                    y=allocated_memory,
                    name=f"{model} allocated",
                    mode="lines+markers"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            title="Memory Usage Analysis",
            xaxis_title="Batch Size",
            xaxis2_title="Batch Size",
            yaxis_title="Memory (MB)",
            yaxis2_title="Memory (MB)"
        )
        
        return fig

    def _plot_hardware_utilization(
        self,
        utilization: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Plot hardware utilization analysis."""
        fig = go.Figure()
        
        models = list(utilization.keys())
        gpu_utils = [utilization[m].get("gpu", 0) for m in models]
        memory_utils = [utilization[m].get("memory", 0) for m in models]
        
        fig.add_trace(go.Bar(
            name="GPU Utilization",
            x=models,
            y=gpu_utils,
            text=gpu_utils,
            textposition="auto"
        ))
        
        fig.add_trace(go.Bar(
            name="Memory Utilization",
            x=models,
            y=memory_utils,
            text=memory_utils,
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Hardware Utilization Analysis",
            barmode="group",
            yaxis_title="Utilization (%)"
        )
        
        return fig

    def analyze_training_dynamics(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_epochs: int = 10
    ) -> Dict[str, Any]:
        """Analyze training dynamics during distillation.
        
        Tracks and analyzes:
        - Loss evolution
        - Accuracy progression
        - Gradient behavior
        - Weight statistics
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model for distillation
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computation device
            n_epochs: Number of epochs to analyze
            
        Returns:
            Dict containing:
                - train: Training metrics history
                - val: Validation metrics history
                - gradients: Gradient statistics
                - weights: Weight statistics
                
        Example:
            >>> dynamics = analyzer.analyze_training_dynamics(
            ...     student, teacher, train_loader, val_loader, device
            ... )
            >>> train_loss = dynamics["train"]["loss"]
        """
        dynamics = {
            "train": {"loss": [], "accuracy": [], "agreement": []},
            "val": {"loss": [], "accuracy": [], "agreement": []},
            "gradients": [],
            "weights": []
        }
        
        optimizer = torch.optim.Adam(student_model.parameters())
        
        for epoch in range(n_epochs):
            # Training phase
            train_metrics = self._train_epoch(
                student_model,
                teacher_model,
                train_loader,
                optimizer,
                device
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(
                student_model,
                teacher_model,
                val_loader,
                device
            )
            
            # Record metrics
            for phase, metrics in [("train", train_metrics), ("val", val_metrics)]:
                dynamics[phase]["loss"].append(metrics["loss"])
                dynamics[phase]["accuracy"].append(metrics["accuracy"])
                dynamics[phase]["agreement"].append(metrics["agreement"])
            
            # Record gradient and weight statistics
            grad_stats = self._compute_gradient_statistics(student_model)
            weight_stats = self._compute_weight_statistics(student_model)
            
            dynamics["gradients"].append(grad_stats)
            dynamics["weights"].append(weight_stats)
        
        return dynamics

    def _train_epoch(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Dict[str, float]:
        """Train for one epoch and collect metrics."""
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0
        correct = 0
        agreement = 0
        total = 0
        
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            student_outputs = student_model(data)
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            # Compute loss
            loss = self._compute_distillation_loss(
                student_outputs,
                teacher_outputs,
                labels
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            total_loss += loss.item()
            student_preds = student_outputs.argmax(dim=1)
            teacher_preds = teacher_outputs.argmax(dim=1)
            
            correct += (student_preds == labels).sum().item()
            agreement += (student_preds == teacher_preds).sum().item()
            total += labels.size(0)
        
        return {
            "loss": total_loss / len(data_loader),
            "accuracy": correct / total,
            "agreement": agreement / total
        }

    def _validate_epoch(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Validate and collect metrics."""
        student_model.eval()
        teacher_model.eval()
        
        total_loss = 0
        correct = 0
        agreement = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Forward pass
                student_outputs = student_model(data)
                teacher_outputs = teacher_model(data)
                
                # Compute loss
                loss = self._compute_distillation_loss(
                    student_outputs,
                    teacher_outputs,
                    labels
                )
                
                # Compute metrics
                total_loss += loss.item()
                student_preds = student_outputs.argmax(dim=1)
                teacher_preds = teacher_outputs.argmax(dim=1)
                
                correct += (student_preds == labels).sum().item()
                agreement += (student_preds == teacher_preds).sum().item()
                total += labels.size(0)
        
        return {
            "loss": total_loss / len(data_loader),
            "accuracy": correct / total,
            "agreement": agreement / total
        }

    def _compute_gradient_statistics(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Compute gradient statistics."""
        stats = {
            "norm": [],
            "mean": [],
            "std": [],
            "zero_fraction": []
        }
        
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                stats["norm"].append(float(grad.norm().item()))
                stats["mean"].append(float(grad.mean().item()))
                stats["std"].append(float(grad.std().item()))
                stats["zero_fraction"].append(
                    float((grad == 0).float().mean().item())
                )
        
        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v))
            }
            for k, v in stats.items()
        }

    def _compute_weight_statistics(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Compute weight statistics."""
        stats = {
            "norm": [],
            "mean": [],
            "std": [],
            "sparsity": []
        }
        
        for param in model.parameters():
            if param.dim() > 1:  # Skip biases
                stats["norm"].append(float(param.norm().item()))
                stats["mean"].append(float(param.mean().item()))
                stats["std"].append(float(param.std().item()))
                stats["sparsity"].append(
                    float((param == 0).float().mean().item())
                )
        
        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v))
            }
            for k, v in stats.items()
        }

    def plot_training_dynamics(
        self,
        dynamics: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Plot training dynamics analysis."""
        plots = {}
        
        # Plot metrics
        plots["metrics"] = self._plot_training_metrics(dynamics)
        
        # Plot gradient statistics
        plots["gradients"] = self._plot_gradient_statistics(dynamics["gradients"])
        
        # Plot weight statistics
        plots["weights"] = self._plot_weight_statistics(dynamics["weights"])
        
        return plots

    def _plot_training_metrics(
        self,
        dynamics: Dict[str, Any]
    ) -> go.Figure:
        """Plot training metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss',
                'Accuracy',
                'Teacher Agreement',
                'Train vs Val Loss'
            )
        )
        
        epochs = list(range(len(dynamics["train"]["loss"])))
        
        # Plot loss
        for phase in ["train", "val"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=dynamics[phase]["loss"],
                    name=f"{phase} loss",
                    mode="lines+markers"
                ),
                row=1, col=1
            )
        
        # Plot accuracy
        for phase in ["train", "val"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=dynamics[phase]["accuracy"],
                    name=f"{phase} accuracy",
                    mode="lines+markers"
                ),
                row=1, col=2
            )
        
        # Plot agreement
        for phase in ["train", "val"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=dynamics[phase]["agreement"],
                    name=f"{phase} agreement",
                    mode="lines+markers"
                ),
                row=2, col=1
            )
        
        # Plot train vs val loss
        fig.add_trace(
            go.Scatter(
                x=dynamics["train"]["loss"],
                y=dynamics["val"]["loss"],
                mode="markers",
                name="loss correlation"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Training Dynamics",
            showlegend=True
        )
        
        return fig

    def _plot_gradient_statistics(
        self,
        gradient_stats: List[Dict[str, Dict[str, float]]]
    ) -> go.Figure:
        """Plot gradient statistics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Gradient Norm',
                'Gradient Mean',
                'Gradient Std',
                'Zero Fraction'
            )
        )
        
        epochs = list(range(len(gradient_stats)))
        metrics = ["norm", "mean", "std", "zero_fraction"]
        
        for i, metric in enumerate(metrics, 1):
            values = [stats[metric]["mean"] for stats in gradient_stats]
            stds = [stats[metric]["std"] for stats in gradient_stats]
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=metric,
                    mode="lines+markers",
                    error_y=dict(
                        type="data",
                        array=stds,
                        visible=True
                    )
                ),
                row=(i-1)//2 + 1,
                col=(i-1)%2 + 1
            )
        
        fig.update_layout(
            height=800,
            title="Gradient Statistics",
            showlegend=True
        )
        
        return fig

    def _plot_weight_statistics(
        self,
        weight_stats: List[Dict[str, Dict[str, float]]]
    ) -> go.Figure:
        """Plot weight statistics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Weight Norm',
                'Weight Mean',
                'Weight Std',
                'Weight Sparsity'
            )
        )
        
        epochs = list(range(len(weight_stats)))
        metrics = ["norm", "mean", "std", "sparsity"]
        
        for i, metric in enumerate(metrics, 1):
            values = [stats[metric]["mean"] for stats in weight_stats]
            stds = [stats[metric]["std"] for stats in weight_stats]
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=metric,
                    mode="lines+markers",
                    error_y=dict(
                        type="data",
                        array=stds,
                        visible=True
                    )
                ),
                row=(i-1)//2 + 1,
                col=(i-1)%2 + 1
            )
        
        fig.update_layout(
            height=800,
            title="Weight Statistics",
            showlegend=True
        )
        
        return fig

    def analyze_adaptively_with_early_stopping(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        patience: int = 5,
        min_epochs: int = 10,
        max_epochs: int = 100,
        target_metrics: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Analyze training with adaptive early stopping."""
        if target_metrics is None:
            target_metrics = {
                "accuracy": 0.9,
                "agreement": 0.8,
                "loss": 0.1
            }
        
        dynamics = {
            "train": {"loss": [], "accuracy": [], "agreement": []},
            "val": {"loss": [], "accuracy": [], "agreement": []},
            "gradients": [],
            "weights": [],
            "early_stop": {
                "stopped_epoch": -1,
                "best_epoch": -1,
                "reason": None
            }
        }
        
        optimizer = torch.optim.Adam(student_model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3
        )
        
        best_val_loss = float('inf')
        best_epoch = -1
        no_improve_count = 0
        
        for epoch in range(max_epochs):
            # Training phase
            train_metrics = self._train_epoch_with_adaptation(
                student_model,
                teacher_model,
                train_loader,
                optimizer,
                device,
                epoch
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(
                student_model,
                teacher_model,
                val_loader,
                device
            )
            
            # Update learning rate
            scheduler.step(val_metrics["loss"])
            
            # Record metrics
            for phase, metrics in [("train", train_metrics), ("val", val_metrics)]:
                dynamics[phase]["loss"].append(metrics["loss"])
                dynamics[phase]["accuracy"].append(metrics["accuracy"])
                dynamics[phase]["agreement"].append(metrics["agreement"])
            
            # Record statistics
            grad_stats = self._compute_gradient_statistics(student_model)
            weight_stats = self._compute_weight_statistics(student_model)
            dynamics["gradients"].append(grad_stats)
            dynamics["weights"].append(weight_stats)
            
            # Check early stopping conditions
            if epoch >= min_epochs:
                # Check target metrics
                targets_met = all(
                    val_metrics[metric] >= target
                    for metric, target in target_metrics.items()
                    if metric in val_metrics
                )
                if targets_met:
                    dynamics["early_stop"].update({
                        "stopped_epoch": epoch,
                        "best_epoch": epoch,
                        "reason": "targets_met"
                    })
                    break
                
                # Check for improvement
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_epoch = epoch
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check patience
                if no_improve_count >= patience:
                    dynamics["early_stop"].update({
                        "stopped_epoch": epoch,
                        "best_epoch": best_epoch,
                        "reason": "no_improvement"
                    })
                    break
        
        return dynamics

    def _train_epoch_with_adaptation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with adaptive mechanisms."""
        student_model.train()
        teacher_model.eval()
        
        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "agreement": 0.0,
            "gradient_norm": 0.0
        }
        
        total = 0
        correct = 0
        agreement = 0
        
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            student_outputs = student_model(data)
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            # Compute adaptive loss
            loss = self._compute_adaptive_loss(
                student_outputs,
                teacher_outputs,
                labels,
                epoch,
                batch_idx
            )
            
            # Backward pass with gradient clipping
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            
            # Update metrics
            metrics["loss"] += loss.item()
            metrics["gradient_norm"] += grad_norm.item()
            
            student_preds = student_outputs.argmax(dim=1)
            teacher_preds = teacher_outputs.argmax(dim=1)
            
            correct += (student_preds == labels).sum().item()
            agreement += (student_preds == teacher_preds).sum().item()
            total += labels.size(0)
        
        # Compute final metrics
        metrics["loss"] /= len(data_loader)
        metrics["gradient_norm"] /= len(data_loader)
        metrics["accuracy"] = correct / total
        metrics["agreement"] = agreement / total
        
        return metrics

    def _compute_adaptive_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        batch_idx: int
    ) -> torch.Tensor:
        """Compute loss with adaptive weights."""
        # Compute standard losses
        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / self.config.temperature, dim=1),
            F.softmax(teacher_outputs / self.config.temperature, dim=1),
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_outputs, labels)
        
        # Compute adaptive weights
        progress = epoch + batch_idx / 1000  # Approximate training progress
        soft_weight = self._compute_soft_weight(progress)
        hard_weight = 1.0 - soft_weight
        
        # Combine losses
        total_loss = soft_weight * soft_loss + hard_weight * hard_loss
        
        return total_loss

    def _compute_soft_weight(
        self,
        progress: float,
        min_weight: float = 0.1,
        max_weight: float = 0.9
    ) -> float:
        """Compute adaptive soft target weight."""
        # Decrease soft target weight over time
        weight = max_weight - (max_weight - min_weight) * min(progress / 50.0, 1.0)
        return float(weight)

    # Add these methods to QualityAnalyzer class

    def analyze_model_compression(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze model compression characteristics.
        
        Evaluates multiple aspects of model compression:
        - Model size reduction
        - Computational complexity
        - Parameter distribution
        - Quantization potential
        
        Args:
            student_model: Compressed student model
            teacher_model: Original teacher model
            device: Computation device
            
        Returns:
            Dict containing:
                - size: Model size analysis
                - complexity: Computational complexity analysis
                - parameters: Parameter distribution analysis
                - compression_score: Overall compression quality score
                
        Example:
            >>> compression = analyzer.analyze_model_compression(student, teacher, device)
            >>> print(f"Compression ratio: {compression['size']['parameters']['ratio']}")
        """
        compression = {}
        
        # Analyze model size
        compression["size"] = self._analyze_model_size(
            student_model, teacher_model
        )
        
        # Analyze computational complexity
        compression["complexity"] = self._analyze_computational_complexity(
            student_model, teacher_model
        )
        
        # Analyze parameter distribution
        compression["parameters"] = self._analyze_parameter_distribution(
            student_model, teacher_model
        )
        
        # Compute compression score
        compression["compression_score"] = self._compute_compression_score(
            compression
        )
        
        return compression

    def _analyze_model_size(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze model size characteristics."""
        size = {}
        
        # Count parameters
        student_params = sum(p.numel() for p in student_model.parameters())
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        
        size["parameters"] = {
            "student": student_params,
            "teacher": teacher_params,
            "ratio": student_params / teacher_params
        }
        
        # Analyze layer-wise sizes
        size["layers"] = self._analyze_layer_sizes(
            student_model, teacher_model
        )
        
        # Analyze model structure
        size["structure"] = {
            "student": self._analyze_model_structure(student_model),
            "teacher": self._analyze_model_structure(teacher_model)
        }
        
        return size

    def _analyze_computational_complexity(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze computational complexity."""
        complexity = {}
        
        # Compute FLOPs
        student_flops = self._compute_model_flops(student_model)
        teacher_flops = self._compute_model_flops(teacher_model)
        
        complexity["flops"] = {
            "student": student_flops,
            "teacher": teacher_flops,
            "ratio": student_flops / teacher_flops
        }
        
        # Analyze layer-wise complexity
        complexity["layers"] = self._analyze_layer_complexity(
            student_model, teacher_model
        )
        
        return complexity

    def _analyze_parameter_distribution(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze parameter distribution characteristics."""
        distribution = {}
        
        # Analyze weight distributions
        distribution["weights"] = {
            "student": self._analyze_weight_distribution(student_model),
            "teacher": self._analyze_weight_distribution(teacher_model)
        }
        
        # Analyze sparsity
        distribution["sparsity"] = {
            "student": self._compute_model_sparsity(student_model),
            "teacher": self._compute_model_sparsity(teacher_model)
        }
        
        # Analyze quantization potential
        distribution["quantization"] = self._analyze_quantization_potential(
            student_model, teacher_model
        )
        
        return distribution

    def _analyze_layer_sizes(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze layer-wise size characteristics."""
        layer_sizes = {
            "student": [],
            "teacher": []
        }
        
        for model_name, model in [
            ("student", student_model),
            ("teacher", teacher_model)
        ]:
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    params = sum(p.numel() for p in module.parameters())
                    layer_sizes[model_name].append({
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": params,
                        "shape": [p.shape for p in module.parameters()]
                    })
        
        return layer_sizes

    def _analyze_model_structure(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze model structure characteristics."""
        structure = {
            "depth": 0,
            "width": [],
            "layer_types": {},
            "connectivity": {}
        }
        
        # Analyze depth and width
        current_depth = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                current_depth += 1
                if isinstance(module, nn.Conv2d):
                    structure["width"].append(module.out_channels)
                elif isinstance(module, nn.Linear):
                    structure["width"].append(module.out_features)
        
        structure["depth"] = current_depth
        
        # Count layer types
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            if layer_type not in structure["layer_types"]:
                structure["layer_types"][layer_type] = 0
            structure["layer_types"][layer_type] += 1
        
        return structure

    def _compute_model_flops(
        self,
        model: nn.Module
    ) -> int:
        """Compute approximate FLOPs for the model."""
        total_flops = 0
        
        def conv2d_flops(module: nn.Conv2d, input_shape: torch.Size) -> int:
            batch_size, in_channels, h, w = input_shape
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            output_h = (h + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
            output_w = (w + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1
            return batch_size * out_channels * output_h * output_w * in_channels * kernel_size
        
        def linear_flops(module: nn.Linear, input_shape: torch.Size) -> int:
            batch_size = input_shape[0]
            return batch_size * module.in_features * module.out_features
        
        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                total_flops += conv2d_flops(module, input[0].shape)
            elif isinstance(module, nn.Linear):
                total_flops += linear_flops(module, input[0].shape)
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass with dummy input
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            model(dummy_input)
        finally:
            for hook in hooks:
                hook.remove()
        
        return total_flops

    def _analyze_layer_complexity(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze layer-wise computational complexity.
        
        Computes:
        - FLOPs per layer
        - Memory requirements
        - Operation types
        - Layer-wise complexity ratios
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            
        Returns:
            Dict containing layer-wise complexity analysis for both models
        """
        complexity = {
            "student": [],
            "teacher": []
        }
        
        for model_name, model in [
            ("student", student_model),
            ("teacher", teacher_model)
        ]:
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    layer_flops = 0
                    if isinstance(module, nn.Conv2d):
                        # Compute Conv2d FLOPs
                        in_channels = module.in_channels
                        out_channels = module.out_channels
                        kernel_size = module.kernel_size[0] * module.kernel_size[1]
                        output_size = 224  # Assuming 224x224 input
                        layer_flops = out_channels * kernel_size * in_channels * output_size * output_size
                    elif isinstance(module, nn.Linear):
                        # Compute Linear FLOPs
                        layer_flops = module.in_features * module.out_features
                    
                    complexity[model_name].append({
                        "name": name,
                        "type": type(module).__name__,
                        "flops": layer_flops
                    })
        
        return complexity

    def _analyze_weight_distribution(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze distribution of model weights.
        
        Computes:
        - Basic statistics (mean, std, range)
        - Weight histograms
        - Quartile ranges
        - Layer-wise distributions
        
        Args:
            model: Model to analyze weights
            
        Returns:
            Dict containing:
                - statistics: Basic statistical measures
                - histograms: Weight distribution histograms
                - ranges: Quartile ranges per layer
        """
        distribution = {
            "statistics": {},
            "histograms": {},
            "ranges": {}
        }
        
        all_weights = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weights = param.detach().cpu().numpy().flatten()
                all_weights.extend(weights)
                
                # Compute statistics
                distribution["statistics"][name] = {
                    "mean": float(np.mean(weights)),
                    "std": float(np.std(weights)),
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights))
                }
                
                # Compute histogram
                hist, bins = np.histogram(weights, bins=50)
                distribution["histograms"][name] = {
                    "counts": hist.tolist(),
                    "bins": bins.tolist()
                }
                
                # Compute ranges
                distribution["ranges"][name] = {
                    "q25": float(np.percentile(weights, 25)),
                    "q50": float(np.percentile(weights, 50)),
                    "q75": float(np.percentile(weights, 75))
                }
        
        # Compute overall statistics
        if all_weights:
            distribution["overall"] = {
                "mean": float(np.mean(all_weights)),
                "std": float(np.std(all_weights)),
                "min": float(np.min(all_weights)),
                "max": float(np.max(all_weights))
            }
        
        return distribution

    def _compute_model_sparsity(
        self,
        model: nn.Module,
        threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """Compute sparsity characteristics of model parameters.
        
        Analyzes:
        - Overall sparsity ratio
        - Layer-wise sparsity
        - Zero-value distributions
        - Threshold-based sparsity
        
        Args:
            model: Model to analyze sparsity
            threshold: Value threshold for considering parameter as zero
            
        Returns:
            Dict containing:
                - overall: Overall model sparsity ratio
                - layers: Layer-wise sparsity analysis
                - statistics: Sparsity statistics
        """
        sparsity = {
            "overall": 0.0,
            "layers": {}
        }
        
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if "weight" in name:
                param_array = param.detach().cpu().numpy()
                total = param_array.size
                zeros = np.sum(np.abs(param_array) < threshold)
                
                sparsity["layers"][name] = float(zeros / total)
                
                total_params += total
                zero_params += zeros
        
        if total_params > 0:
            sparsity["overall"] = float(zero_params / total_params)
        
        return sparsity

    def _analyze_quantization_potential(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze potential for model quantization.
        
        Evaluates:
        - Dynamic range requirements
        - Weight clustering patterns
        - Precision requirements
        - Quantization impact estimates
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            
        Returns:
            Dict containing:
                - student/teacher: Per-model quantization metrics
                - comparison: Comparative analysis
                - requirements: Precision requirements
        """
        potential = {
            "student": self._compute_quantization_metrics(student_model),
            "teacher": self._compute_quantization_metrics(teacher_model)
        }
        
        # Compare student vs teacher
        potential["comparison"] = {
            "dynamic_range_ratio": (
                potential["student"]["dynamic_range"] /
                potential["teacher"]["dynamic_range"]
            ),
            "precision_requirements": {
                "student": self._estimate_precision_requirements(student_model),
                "teacher": self._estimate_precision_requirements(teacher_model)
            }
        }
        
        return potential

    def _compute_quantization_metrics(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Compute metrics related to quantization potential.
        
        Analyzes:
        - Value distributions
        - Dynamic ranges
        - Clustering tendencies
        - Statistical characteristics
        
        Args:
            model: Model to analyze for quantization
            
        Returns:
            Dict containing:
                - dynamic_range: Value range metrics
                - distribution: Distribution characteristics
                - clustering: Clustering analysis
        """
        metrics = {}
        
        all_weights = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weights = param.detach().cpu().numpy().flatten()
                all_weights.extend(weights)
        
        if all_weights:
            weights = np.array(all_weights)
            
            # Compute dynamic range
            metrics["dynamic_range"] = float(np.max(np.abs(weights)))
            
            # Compute distribution characteristics
            metrics["distribution"] = {
                "skewness": float(scipy.stats.skew(weights)),
                "kurtosis": float(scipy.stats.kurtosis(weights))
            }
            
            # Estimate clustering tendency
            metrics["clustering"] = {
                "unique_values": len(np.unique(weights)),
                "unique_ratio": len(np.unique(weights)) / len(weights)
            }
        
        return metrics

    def _estimate_precision_requirements(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Estimate required numerical precision for model parameters.
        
        Analyzes:
        - Value ranges per layer
        - Minimum representable values
        - Required bit widths
        - Dynamic range requirements
        
        Args:
            model: Model to analyze precision requirements
            
        Returns:
            Dict containing per-layer precision requirements:
                - estimated_bits: Estimated bits needed
                - dynamic_range: Dynamic range analysis
        """
        requirements = {}
        
        for name, param in model.named_parameters():
            if "weight" in name:
                weights = param.detach().cpu().numpy().flatten()
                
                # Estimate minimum bits needed
                max_val = np.max(np.abs(weights))
                min_val = np.min(np.abs(weights[weights != 0]))
                
                if max_val > 0 and min_val > 0:
                    dynamic_range = np.log2(max_val / min_val)
                    requirements[name] = {
                        "estimated_bits": int(np.ceil(dynamic_range)) + 1,
                        "dynamic_range": float(dynamic_range)
                    }
        
        return requirements

    def _compute_compression_score(
        self,
        compression: Dict[str, Any]
    ) -> float:
        """Compute overall compression quality score.
        
        Considers:
        - Size reduction ratio
        - Computational efficiency
        - Parameter efficiency
        - Quantization potential
        
        Args:
            compression: Compression analysis results
            
        Returns:
            Float between 0 and 1 indicating compression quality
        """
        scores = []
        
        # Score size reduction
        if "size" in compression:
            size_ratio = compression["size"]["parameters"]["ratio"]
            scores.append(1.0 - min(size_ratio, 1.0))
        
        # Score computational efficiency
        if "complexity" in compression:
            flops_ratio = compression["complexity"]["flops"]["ratio"]
            scores.append(1.0 - min(flops_ratio, 1.0))
        
        # Score parameter efficiency
        if "parameters" in compression:
            sparsity = compression["parameters"]["sparsity"]["student"]["overall"]
            scores.append(sparsity)
            
            # Score quantization potential
            quant = compression["parameters"]["quantization"]["student"]
            if "clustering" in quant:
                cluster_score = 1.0 - quant["clustering"]["unique_ratio"]
                scores.append(cluster_score)
        
        return float(np.mean(scores)) if scores else 0.0

    # Add these methods to QualityAnalyzer class

    def optimize_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_trials: int = 50,
        n_epochs: int = 100,
        patience: int = 10
    ) -> Dict[str, Any]:
        """Optimize distillation process with automated hyperparameter tuning.
        
        Uses Optuna for hyperparameter optimization with early stopping.
        Optimizes temperature, alpha, learning rate, batch size, etc.
        
        Args:
            student_model: Student model to optimize
            teacher_model: Teacher model for distillation
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computation device
            n_trials: Number of optimization trials
            n_epochs: Maximum epochs per trial
            patience: Early stopping patience
            
        Returns:
            Dict containing:
                - best_config: Best hyperparameters found
                - best_score: Best optimization score
                - optimization_history: Full optimization history
                - final_results: Results with best configuration
                
        Example:
            >>> opt_results = analyzer.optimize_distillation(
            ...     student, teacher, train_loader, val_loader, device
            ... )
            >>> best_config = opt_results["best_config"]
        """
        from optuna import create_study, Trial
        
        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            config = {
                "temperature": trial.suggest_float("temperature", 1.0, 10.0),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
            }
            
            # Train with early stopping
            results = self.analyze_adaptively_with_early_stopping(
                student_model=student_model,
                teacher_model=teacher_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                patience=patience,
                min_epochs=10,
                max_epochs=n_epochs,
                config=config
            )
            
            # Compute optimization score
            score = self._compute_optimization_score_from_results(results)
            
            return score
        
        # Create and run study
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best configuration
        best_config = study.best_params
        best_score = study.best_value
        
        # Train final model with best config
        final_results = self.analyze_adaptively_with_early_stopping(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            patience=patience,
            min_epochs=10,
            max_epochs=n_epochs,
            config=best_config
        )
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "optimization_history": study.trials_dataframe().to_dict(),
            "final_results": final_results
        }

    def _compute_optimization_score_from_results(
        self,
        results: Dict[str, Any]
    ) -> float:
        """Compute optimization score from training results."""
        scores = []
        
        # Score validation metrics
        if "val" in results:
            val_metrics = results["val"]
            if "accuracy" in val_metrics:
                scores.append(float(val_metrics["accuracy"][-1]))
            if "agreement" in val_metrics:
                scores.append(float(val_metrics["agreement"][-1]))
        
        # Score training stability
        if "train" in results:
            train_metrics = results["train"]
            if "loss" in train_metrics:
                loss_stability = 1.0 - float(np.std(train_metrics["loss"]) / np.mean(train_metrics["loss"]))
                scores.append(loss_stability)
        
        # Score gradient behavior
        if "gradients" in results:
            grad_stats = results["gradients"][-1]
            if "norm" in grad_stats:
                grad_stability = 1.0 - min(grad_stats["norm"]["std"] / grad_stats["norm"]["mean"], 1.0)
                scores.append(grad_stability)
        
        # Score early stopping
        if "early_stop" in results:
            early_stop = results["early_stop"]
            if early_stop["reason"] == "targets_met":
                scores.append(1.0)
            elif early_stop["reason"] == "no_improvement":
                progress = early_stop["stopped_epoch"] / early_stop["best_epoch"]
                scores.append(max(0.0, 1.0 - progress))
        
        return float(np.mean(scores)) if scores else 0.0

    def evaluate_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of distillation results."""
        evaluation = {}
        
        # Basic quality metrics
        evaluation["quality"] = self.analyze_quality(
            student_model,
            teacher_model,
            test_loader,
            device
        )
        
        # Performance analysis
        evaluation["performance"] = self.analyze_performance(
            student_model,
            teacher_model,
            test_loader,
            device
        )
        
        # Compression analysis
        evaluation["compression"] = self.analyze_model_compression(
            student_model,
            teacher_model,
            device
        )
        
        # Prediction behavior
        evaluation["prediction"] = self._analyze_prediction_behavior(
            student_model,
            teacher_model,
            test_loader,
            device
        )
        
        # Generate evaluation report
        evaluation["report"] = self._generate_evaluation_report(evaluation)
        
        return evaluation

    def _generate_evaluation_report(
        self,
        evaluation: Dict[str, Any]
    ) -> str:
        """Generate comprehensive evaluation report."""
        template = """
        # Knowledge Distillation Evaluation Report

        ## Quality Metrics
        {% for metric, value in quality_scores.items() %}
        - {{ metric }}: {{ "%.3f"|format(value) }}
        {% endfor %}

        ## Performance Analysis
        - Inference Speed: {{ "%.2f"|format(performance.inference.speedup) }}x faster
        - Memory Usage: {{ "%.2f"|format(performance.memory.reduction) }}x reduction
        - Hardware Utilization: {{ "%.1f"|format(performance.utilization.efficiency) }}% efficient

        ## Compression Results
        - Model Size: {{ "%.2f"|format(compression.size.ratio) }}x smaller
        - FLOPs: {{ "%.2f"|format(compression.complexity.flops.ratio) }}x reduction
        - Parameter Efficiency: {{ "%.1f"|format(compression.parameters.efficiency) }}%

        ## Prediction Analysis
        - Agreement with Teacher: {{ "%.1f"|format(prediction.agreement_rate * 100) }}%
        - Confidence Calibration: {{ "%.3f"|format(prediction.calibration.score) }}
        - Error Analysis: {{ prediction.error_analysis.summary }}

        ## Recommendations
        {% for rec in recommendations %}
        * {{ rec }}
        {% endfor %}
        """
        
        # Prepare template variables
        template_vars = self._prepare_report_variables(evaluation)
        
        # Generate recommendations
        template_vars["recommendations"] = self._generate_evaluation_recommendations(
            evaluation
        )
        
        # Render template
        env = jinja2.Environment()
        template = env.from_string(template)
        return template.render(**template_vars)

    def _prepare_report_variables(
        self,
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare variables for evaluation report."""
        variables = {
            "quality_scores": {},
            "performance": {
                "inference": {"speedup": 0.0},
                "memory": {"reduction": 0.0},
                "utilization": {"efficiency": 0.0}
            },
            "compression": {
                "size": {"ratio": 0.0},
                "complexity": {"flops": {"ratio": 0.0}},
                "parameters": {"efficiency": 0.0}
            },
            "prediction": {
                "agreement_rate": 0.0,
                "calibration": {"score": 0.0},
                "error_analysis": {"summary": ""}
            }
        }
        
        # Extract quality scores
        if "quality" in evaluation:
            for metric, result in evaluation["quality"].items():
                if isinstance(result, dict) and "score" in result:
                    variables["quality_scores"][metric] = result["score"]
        
        # Extract performance metrics
        if "performance" in evaluation:
            perf = evaluation["performance"]
            if "inference" in perf:
                student_latency = perf["inference"]["student"]["latency"].get(32, 0)
                teacher_latency = perf["inference"]["teacher"]["latency"].get(32, 0)
                if teacher_latency > 0:
                    variables["performance"]["inference"]["speedup"] = (
                        teacher_latency / student_latency
                    )
        
        # Extract compression metrics
        if "compression" in evaluation:
            comp = evaluation["compression"]
            if "size" in comp:
                variables["compression"]["size"]["ratio"] = (
                    1.0 / comp["size"]["parameters"]["ratio"]
                )
        
        # Extract prediction metrics
        if "prediction" in evaluation:
            pred = evaluation["prediction"]
            if "diversity" in pred:
                variables["prediction"]["agreement_rate"] = (
                    pred["diversity"].get("agreement_rate", 0.0)
                )
        
        return variables

    def _generate_evaluation_recommendations(
        self,
        evaluation: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Quality-based recommendations
        if "quality" in evaluation:
            quality = evaluation["quality"]
            if quality.get("retention_score", 1.0) < 0.8:
                recommendations.append(
                    "Consider increasing temperature or using intermediate layers"
                )
        
        # Performance-based recommendations
        if "performance" in evaluation:
            perf = evaluation["performance"]
            if "inference" in perf:
                speedup = perf["inference"]["student"]["latency"].get(32, 0)
                if speedup < 2.0:
                    recommendations.append(
                        "Consider further model compression or architecture optimization"
                    )
        
        # Compression-based recommendations
        if "compression" in evaluation:
            comp = evaluation["compression"]
            if "parameters" in comp:
                if comp["parameters"]["sparsity"]["student"]["overall"] < 0.5:
                    recommendations.append(
                        "Consider applying pruning or quantization"
                    )
        
        # Prediction-based recommendations
        if "prediction" in evaluation:
            pred = evaluation["prediction"]
            if "diversity" in pred:
                if pred["diversity"].get("agreement_rate", 1.0) < 0.7:
                    recommendations.append(
                        "Consider adjusting distillation loss weights"
                    )
        
        return recommendations

    # Add these methods to QualityAnalyzer class

    def generate_interactive_report(
        self,
        evaluation: Dict[str, Any],
        output_dir: str
    ) -> str:
        """Generate interactive HTML report with plotly visualizations.
        
        Creates a comprehensive report with:
        - Interactive plots
        - Quality metrics
        - Performance analysis
        - Compression analysis
        - Prediction behavior
        
        Args:
            evaluation: Evaluation results from analyze_quality()
            output_dir: Directory to save report and plots
            
        Returns:
            Path to generated HTML report
            
        Example:
            >>> results = analyzer.analyze_quality(student, teacher, test_loader, device)
            >>> report_path = analyzer.generate_interactive_report(results, "reports/")
        """
        # Create report directory
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plots = {
            "quality": self.plot_quality_metrics(evaluation["quality"]),
            "performance": self.plot_performance_analysis(evaluation["performance"]),
            "compression": self._plot_compression_analysis(evaluation["compression"]),
            "prediction": self._plot_prediction_behavior(evaluation["prediction"])
        }
        
        # Save plots as HTML files
        plot_files = {}
        for category, category_plots in plots.items():
            category_dir = report_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for name, fig in category_plots.items():
                file_path = category_dir / f"{name}.html"
                fig.write_html(str(file_path))
                plot_files[f"{category}_{name}"] = str(file_path)
        
        # Generate report content
        report_content = self._generate_interactive_report_content(
            evaluation,
            plot_files
        )
        
        # Save report
        report_path = report_dir / "interactive_report.html"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return str(report_path)

    def _plot_compression_analysis(
        self,
        compression: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Generate compression analysis plots."""
        plots = {}
        
        # Plot model size comparison
        plots["size"] = self._plot_model_size_comparison(
            compression["size"]
        )
        
        # Plot computational complexity
        plots["complexity"] = self._plot_computational_complexity(
            compression["complexity"]
        )
        
        # Plot parameter distribution
        plots["parameters"] = self._plot_parameter_distribution(
            compression["parameters"]
        )
        
        return plots

    def _plot_model_size_comparison(
        self,
        size_info: Dict[str, Any]
    ) -> go.Figure:
        """Plot model size comparison."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Parameter Count',
                'Layer-wise Parameters',
                'Model Structure',
                'Size Reduction'
            )
        )
        
        # Plot parameter count
        params = size_info["parameters"]
        fig.add_trace(
            go.Bar(
                x=["Student", "Teacher"],
                y=[params["student"], params["teacher"]],
                name="Parameters"
            ),
            row=1, col=1
        )
        
        # Plot layer-wise parameters
        layers = size_info["layers"]
        for model in ["student", "teacher"]:
            layer_params = [l["parameters"] for l in layers[model]]
            layer_names = [l["name"] for l in layers[model]]
            
            fig.add_trace(
                go.Bar(
                    x=layer_names,
                    y=layer_params,
                    name=f"{model} layers"
                ),
                row=1, col=2
            )
        
        # Plot model structure
        structure = size_info["structure"]
        for model in ["student", "teacher"]:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(structure[model]["width"]))),
                    y=structure[model]["width"],
                    name=f"{model} width",
                    mode="lines+markers"
                ),
                row=2, col=1
            )
        
        # Plot size reduction metrics
        reduction_metrics = {
            "Parameter Ratio": params["ratio"],
            "Effective Compression": 1.0 / params["ratio"]
        }
        
        fig.add_trace(
            go.Bar(
                x=list(reduction_metrics.keys()),
                y=list(reduction_metrics.values()),
                name="Reduction"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Model Size Analysis",
            showlegend=True
        )
        
        return fig

    def _generate_interactive_report_content(
        self,
        evaluation: Dict[str, Any],
        plot_files: Dict[str, str]
    ) -> str:
        """Generate interactive HTML report content."""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Knowledge Distillation Analysis</title>
            <style>
                body { 
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .section {
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .plot-container {
                    margin: 20px 0;
                    height: 600px;
                }
                .metric {
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }
                .metric-label {
                    font-size: 14px;
                    color: #6c757d;
                }
                .recommendations {
                    background-color: #e9ecef;
                    padding: 15px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Knowledge Distillation Analysis Report</h1>
                
                <div class="section">
                    <h2>Key Metrics</h2>
                    {% for metric, value in quality_scores.items() %}
                    <div class="metric">
                        <div class="metric-value">{{ "%.3f"|format(value) }}</div>
                        <div class="metric-label">{{ metric }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Quality Analysis</h2>
                    {% for name, path in quality_plots.items() %}
                    <div class="plot-container">
                        <iframe src="{{ path }}" width="100%" height="100%" frameborder="0"></iframe>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Performance Analysis</h2>
                    {% for name, path in performance_plots.items() %}
                    <div class="plot-container">
                        <iframe src="{{ path }}" width="100%" height="100%" frameborder="0"></iframe>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Compression Analysis</h2>
                    {% for name, path in compression_plots.items() %}
                    <div class="plot-container">
                        <iframe src="{{ path }}" width="100%" height="100%" frameborder="0"></iframe>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Prediction Behavior</h2>
                    {% for name, path in prediction_plots.items() %}
                    <div class="plot-container">
                        <iframe src="{{ path }}" width="100%" height="100%" frameborder="0"></iframe>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendations">
                        {% for rec in recommendations %}
                        <p>{{ rec }}</p>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Prepare template variables
        template_vars = {
            "quality_scores": {
                metric: result["score"]
                for metric, result in evaluation["quality"].items()
                if isinstance(result, dict) and "score" in result
            },
            "quality_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("quality_")
            },
            "performance_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("performance_")
            },
            "compression_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("compression_")
            },
            "prediction_plots": {
                name: path
                for name, path in plot_files.items()
                if name.startswith("prediction_")
            },
            "recommendations": self._generate_evaluation_recommendations(evaluation)
        }
        
        # Render template
        env = jinja2.Environment()
        template = env.from_string(template)
        return template.render(**template_vars)

    # Add these methods to QualityAnalyzer class

    def _plot_computational_complexity(
        self,
        complexity: Dict[str, Any]
    ) -> go.Figure:
        """Plot computational complexity analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'FLOPs Comparison',
                'Layer-wise FLOPs',
                'Complexity Reduction',
                'Operation Distribution'
            )
        )
        
        # Plot FLOPs comparison
        flops = complexity["flops"]
        fig.add_trace(
            go.Bar(
                x=["Student", "Teacher"],
                y=[flops["student"], flops["teacher"]],
                name="FLOPs"
            ),
            row=1, col=1
        )
        
        # Plot layer-wise FLOPs
        layers = complexity["layers"]
        for model in ["student", "teacher"]:
            layer_flops = [l["flops"] for l in layers[model]]
            layer_names = [l["name"] for l in layers[model]]
            
            fig.add_trace(
                go.Bar(
                    x=layer_names,
                    y=layer_flops,
                    name=f"{model} layers"
                ),
                row=1, col=2
            )
        
        # Plot complexity reduction metrics
        reduction_metrics = {
            "FLOPs Ratio": flops["ratio"],
            "Effective Speedup": 1.0 / flops["ratio"]
        }
        
        fig.add_trace(
            go.Bar(
                x=list(reduction_metrics.keys()),
                y=list(reduction_metrics.values()),
                name="Reduction"
            ),
            row=2, col=1
        )
        
        # Plot operation distribution
        op_types = set()
        for model_layers in layers.values():
            for layer in model_layers:
                op_types.add(layer["type"])
        
        op_counts = {
            model: {op: 0 for op in op_types}
            for model in ["student", "teacher"]
        }
        
        for model, model_layers in layers.items():
            for layer in model_layers:
                op_counts[model][layer["type"]] += 1
        
        for model in ["student", "teacher"]:
            fig.add_trace(
                go.Bar(
                    x=list(op_types),
                    y=[op_counts[model][op] for op in op_types],
                    name=f"{model} ops"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Computational Complexity Analysis",
            showlegend=True
        )
        
        return fig

    def _plot_parameter_distribution(
        self,
        distribution: Dict[str, Any]
    ) -> go.Figure:
        """Plot parameter distribution analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Weight Distribution',
                'Sparsity Analysis',
                'Quantization Potential',
                'Layer-wise Statistics'
            )
        )
        
        # Plot weight distribution
        weights = distribution["weights"]
        for model in ["student", "teacher"]:
            stats = weights[model]["statistics"]
            fig.add_trace(
                go.Box(
                    y=list(stats.values()),
                    name=f"{model} weights"
                ),
                row=1, col=1
            )
        
        # Plot sparsity analysis
        sparsity = distribution["sparsity"]
        for model in ["student", "teacher"]:
            layer_sparsity = list(sparsity[model]["layers"].values())
            fig.add_trace(
                go.Box(
                    y=layer_sparsity,
                    name=f"{model} sparsity"
                ),
                row=1, col=2
            )
        
        # Plot quantization potential
        quant = distribution["quantization"]
        metrics = ["dynamic_range", "clustering"]
        for model in ["student", "teacher"]:
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=[quant[model][m] for m in metrics],
                    name=f"{model} quant"
                ),
                row=2, col=1
            )
        
        # Plot layer-wise statistics
        for model in ["student", "teacher"]:
            stats = weights[model]["statistics"]
            fig.add_trace(
                go.Scatter(
                    x=list(stats.keys()),
                    y=[s["mean"] for s in stats.values()],
                    name=f"{model} mean",
                    mode="lines+markers"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Parameter Distribution Analysis",
            showlegend=True
        )
        
        return fig

    def _plot_error_patterns(
        self,
        error_analysis: Dict[str, Any]
    ) -> go.Figure:
        """Plot error pattern analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confusion Matrix Comparison',
                'Error Type Distribution',
                'Class-wise Error Rates',
                'Error Correlation'
            )
        )
        
        # Plot confusion matrices
        confusion = error_analysis["confusion"]
        for i, (model, matrix) in enumerate(confusion.items(), 1):
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    colorscale="RdBu",
                    name=f"{model} confusion"
                ),
                row=1, col=i
            )
        
        # Plot error types
        error_types = error_analysis["error_types"]
        fig.add_trace(
            go.Bar(
                x=list(error_types.keys()),
                y=list(error_types.values()),
                name="Error types"
            ),
            row=2, col=1
        )
        
        # Plot class-wise errors
        class_errors = error_analysis["class_errors"]
        classes = list(class_errors.keys())
        student_errors = [e["student_error"] for e in class_errors.values()]
        teacher_errors = [e["teacher_error"] for e in class_errors.values()]
        
        fig.add_trace(
            go.Bar(
                x=classes,
                y=student_errors,
                name="Student errors"
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=classes,
                y=teacher_errors,
                name="Teacher errors"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Error Pattern Analysis",
            showlegend=True,
            barmode="group"
        )
        
        return fig

    def _analyze_feature_maps(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data: torch.Tensor,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze feature map characteristics between student and teacher.
        
        Compares feature map properties to understand knowledge transfer at
        different network levels.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            data: Input data batch
            device: Computation device
            
        Returns:
            Dict containing:
                - spatial: Spatial characteristics comparison
                - channel: Channel-wise activation analysis
                - statistics: Feature statistics comparison
                - correlation: Feature correlation analysis
        """

    def _analyze_loss_landscape(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze loss landscape around current model state.
        
        Explores loss surface characteristics to understand optimization
        behavior and stability.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for reference
            data: Input data batch
            labels: Ground truth labels
            device: Computation device
            
        Returns:
            Dict containing:
                - surface: Loss surface characteristics
                - gradients: Gradient-based analysis
                - curvature: Local curvature analysis
                - stability: Stability metrics
        """

    def _analyze_hardware_utilization(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze hardware resource utilization.
        
        Measures hardware efficiency and resource usage patterns
        during model execution.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            device: Computation device
            
        Returns:
            Dict containing:
                - memory: Memory usage analysis
                - compute: Computation utilization
                - bandwidth: Memory bandwidth usage
                - efficiency: Hardware efficiency metrics
        """

    def _analyze_training_stability(
        self,
        train_history: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze training stability from history.
        
        Examines training metrics to assess convergence quality
        and stability issues.
        
        Args:
            train_history: List of training metrics per iteration
            
        Returns:
            Dict containing:
                - convergence: Convergence analysis
                - oscillation: Oscillation detection
                - stability_score: Overall stability score
                - issues: Identified stability issues
        """

    def _plot_feature_maps(
        self,
        feature_analysis: Dict[str, Any]
    ) -> go.Figure:
        """Plot feature map analysis results.
        
        Creates visualizations comparing feature map characteristics
        between student and teacher models.
        
        Args:
            feature_analysis: Feature map analysis results
            
        Returns:
            Plotly figure containing:
                - Spatial activation patterns
                - Channel importance comparison
                - Feature statistics visualization
                - Correlation analysis plots
        """

    def _plot_loss_landscape(
        self,
        landscape_analysis: Dict[str, Any]
    ) -> go.Figure:
        """Plot loss landscape analysis.
        
        Visualizes loss surface characteristics and optimization
        behavior analysis.
        
        Args:
            landscape_analysis: Loss landscape analysis results
            
        Returns:
            Plotly figure containing:
                - Loss surface contours
                - Gradient direction analysis
                - Curvature visualization
                - Stability metric plots
        """

    def _plot_hardware_metrics(
        self,
        hardware_analysis: Dict[str, Any]
    ) -> go.Figure:
        """Plot hardware utilization metrics.
        
        Creates visualizations of hardware resource usage and
        efficiency metrics.
        
        Args:
            hardware_analysis: Hardware utilization analysis results
            
        Returns:
            Plotly figure containing:
                - Memory usage patterns
                - Computation utilization
                - Bandwidth usage plots
                - Efficiency comparisons
        """

    def _plot_training_stability(
        self,
        stability_analysis: Dict[str, Any]
    ) -> go.Figure:
        """Plot training stability analysis.
        
        Visualizes training dynamics and stability metrics
        to identify potential issues.
        
        Args:
            stability_analysis: Training stability analysis results
            
        Returns:
            Plotly figure containing:
                - Convergence plots
                - Oscillation analysis
                - Stability metrics
                - Issue identification
        """

    def analyze_feature_maps(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data: torch.Tensor,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze feature map characteristics between student and teacher.
        
        Performs detailed analysis of feature map properties to understand
        knowledge transfer effectiveness at different network levels.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            data: Input data batch for feature extraction
            device: Computation device
            
        Returns:
            Dict containing:
                - spatial: Spatial characteristics analysis
                    - resolution: Resolution comparison
                    - receptive_field: Receptive field analysis
                - channel: Channel-wise analysis
                    - importance: Channel importance scores
                    - redundancy: Channel redundancy metrics
                - statistics: Feature statistics
                    - mean_std: Mean and std comparisons
                    - range: Value range analysis
                - correlation: Feature correlations
                    - layer_wise: Layer-wise correlations
                    - cross_layer: Cross-layer dependencies
        """

    def analyze_loss_landscape(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
        n_points: int = 20,
        step_size: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze loss landscape around current model state.
        
        Explores the loss surface characteristics by sampling points
        around the current model parameters to understand optimization
        behavior and stability.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for reference
            data: Input data batch
            labels: Ground truth labels
            device: Computation device
            n_points: Number of points to sample
            step_size: Step size for parameter perturbation
            
        Returns:
            Dict containing:
                - surface: Loss surface characteristics
                    - values: Loss values at sampled points
                    - gradients: Gradient directions
                    - hessian: Local curvature estimates
                - stability: Stability metrics
                    - smoothness: Surface smoothness measures
                    - barriers: Energy barrier heights
                    - basins: Local minima analysis
        """

    def analyze_hardware_utilization(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: torch.device,
        batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict[str, Any]:
        """Analyze hardware resource utilization patterns.
        
        Measures and compares hardware efficiency metrics between
        student and teacher models under various conditions.
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            device: Computation device
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dict containing:
                - memory: Memory usage analysis
                    - peak: Peak memory usage
                    - dynamic: Dynamic memory allocation
                    - cache: Cache utilization
                - compute: Computation utilization
                    - gpu: GPU utilization metrics
                    - cpu: CPU utilization metrics
                - bandwidth: Memory bandwidth
                    - read: Memory read bandwidth
                    - write: Memory write bandwidth
                - efficiency: Overall efficiency metrics
                    - throughput: Processing throughput
                    - latency: Processing latency
        """

    def analyze_training_stability(
        self,
        train_history: List[Dict[str, float]],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze training stability from history.
        
        Examines training metrics over time to assess convergence
        quality and identify potential stability issues.
        
        Args:
            train_history: List of training metrics per iteration
            window_size: Window size for moving statistics
            
        Returns:
            Dict containing:
                - convergence: Convergence analysis
                    - rate: Convergence rate metrics
                    - trajectory: Loss trajectory analysis
                - oscillation: Oscillation detection
                    - frequency: Oscillation frequencies
                    - amplitude: Oscillation amplitudes
                - stability: Stability metrics
                    - gradient_norm: Gradient norm statistics
                    - weight_updates: Weight update statistics
                - issues: Identified problems
                    - vanishing_gradients: Vanishing gradient detection
                    - exploding_gradients: Exploding gradient detection
        """

    def plot_feature_maps(
        self,
        feature_analysis: Dict[str, Any],
        max_channels: int = 16
    ) -> go.Figure:
        """Plot feature map analysis results.
        
        Creates comprehensive visualizations comparing feature map
        characteristics between student and teacher models.
        
        Args:
            feature_analysis: Feature map analysis results
            max_channels: Maximum number of channels to plot
            
        Returns:
            Plotly figure containing:
                - Spatial activation patterns
                    - Feature map visualizations
                    - Activation heatmaps
                - Channel importance comparison
                    - Channel-wise statistics
                    - Importance scores
                - Feature statistics
                    - Distribution plots
                    - Statistical comparisons
                - Correlation analysis
                    - Correlation matrices
                    - Dependency graphs
        """

    def plot_loss_landscape(
        self,
        landscape_analysis: Dict[str, Any],
        plot_type: str = '3d'
    ) -> go.Figure:
        """Plot loss landscape analysis results.
        
        Creates visualizations of loss surface characteristics
        and optimization behavior analysis.
        
        Args:
            landscape_analysis: Loss landscape analysis results
            plot_type: Type of plot ('3d' or 'contour')
            
        Returns:
            Plotly figure containing:
                - Loss surface visualization
                    - 3D surface or contour plot
                    - Critical points
                - Gradient analysis
                    - Gradient vector field
                    - Gradient magnitude heatmap
                - Stability visualization
                    - Energy barriers
                    - Basin structure
                - Trajectory analysis
                    - Optimization path
                    - Local minima
        """

    def _analyze_knowledge_transfer(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze knowledge transfer effectiveness between teacher and student.
        
        Evaluates how well knowledge is transferred across different aspects:
        - Feature-level transfer
        - Decision boundary alignment
        - Response pattern matching
        - Dark knowledge utilization
        
        Args:
            student_model: Student model receiving knowledge
            teacher_model: Teacher model transferring knowledge
            data_loader: Data loader for evaluation
            device: Computation device
            
        Returns:
            Dict containing:
                - feature_transfer: Feature-level transfer analysis
                    - layer_alignment: Layer-wise alignment scores
                    - feature_matching: Feature matching metrics
                - decision_boundary: Decision boundary analysis
                    - boundary_alignment: Boundary alignment scores
                    - margin_distribution: Margin distribution stats
                - response_patterns: Response pattern analysis
                    - pattern_matching: Pattern matching scores
                    - activation_similarity: Activation similarities
                - dark_knowledge: Dark knowledge analysis
                    - soft_target_usage: Soft target utilization
                    - confidence_transfer: Confidence transfer metrics
        """

    def _analyze_model_robustness(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        perturbation_types: List[str] = ['noise', 'adversarial', 'distribution']
    ) -> Dict[str, Any]:
        """Analyze model robustness under different conditions.
        
        Tests model behavior under various perturbations and conditions:
        - Input perturbations
        - Adversarial attacks
        - Distribution shifts
        - Noise injection
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            data_loader: Data loader for testing
            device: Computation device
            perturbation_types: Types of perturbations to test
            
        Returns:
            Dict containing:
                - noise_robustness: Noise robustness analysis
                    - gaussian_noise: Gaussian noise resistance
                    - impulse_noise: Impulse noise resistance
                - adversarial_robustness: Adversarial attack analysis
                    - fgsm: FGSM attack results
                    - pgd: PGD attack results
                - distribution_robustness: Distribution shift analysis
                    - intensity_shift: Intensity shift results
                    - style_transfer: Style transfer results
                - stability_metrics: Overall stability metrics
                    - consistency: Prediction consistency
                    - sensitivity: Input sensitivity
        """

    def _analyze_optimization_efficiency(
        self,
        train_history: Dict[str, List[float]],
        compute_metrics: Dict[str, float],
        memory_usage: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze optimization efficiency across multiple dimensions.
        
        Evaluates efficiency of the optimization process:
        - Convergence speed
        - Resource utilization
        - Training stability
        - Memory efficiency
        
        Args:
            train_history: Training metrics history
            compute_metrics: Computation resource metrics
            memory_usage: Memory usage statistics
            
        Returns:
            Dict containing:
                - convergence: Convergence efficiency analysis
                    - speed: Convergence speed metrics
                    - stability: Training stability metrics
                - resource_usage: Resource utilization analysis
                    - compute: Computation efficiency
                    - memory: Memory efficiency
                - training_cost: Training cost analysis
                    - time_efficiency: Time-based metrics
                    - resource_efficiency: Resource-based metrics
                - overall_efficiency: Combined efficiency scores
        """

    def _analyze_feature_importance(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze importance of different features in knowledge transfer.
        
        Evaluates which features are most important for:
        - Classification accuracy
        - Knowledge transfer
        - Model compression
        - Generalization
        
        Args:
            student_features: Student model feature maps
            teacher_features: Teacher model feature maps
            labels: Ground truth labels
            
        Returns:
            Dict containing:
                - channel_importance: Channel importance analysis
                    - discriminative: Discriminative power scores
                    - transfer: Transfer importance scores
                - spatial_importance: Spatial importance analysis
                    - attention: Spatial attention maps
                    - receptive_field: Receptive field analysis
                - feature_ranking: Feature importance ranking
                    - classification: Classification importance
                    - distillation: Distillation importance
                - redundancy_analysis: Feature redundancy metrics
        """

    def _analyze_prediction_confidence(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze prediction confidence patterns and calibration.
        
        Examines confidence characteristics:
        - Calibration quality
        - Uncertainty estimation
        - Confidence distribution
        - Agreement patterns
        
        Args:
            student_probs: Student model probabilities (N x C)
            teacher_probs: Teacher model probabilities (N x C)
            true_labels: Ground truth labels (N)
            
        Returns:
            Dict containing:
                - calibration: Calibration analysis
                    - reliability: Reliability diagram data
                    - ece: Expected Calibration Error
                - uncertainty: Uncertainty analysis
                    - entropy: Prediction entropy
                    - mutual_info: Mutual information
                - confidence: Confidence analysis
                    - distribution: Confidence distribution
                    - thresholds: Confidence thresholds
                - agreement: Agreement analysis
                    - patterns: Agreement patterns
                    - correlation: Confidence correlation
        """

    def _analyze_layer_adaptation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze layer adaptation and matching between models.
        
        Examines how well layers are adapted:
        - Architecture alignment
        - Feature transformation
        - Capacity matching
        - Information flow
        
        Args:
            student_model: Student model architecture
            teacher_model: Teacher model architecture
            
        Returns:
            Dict containing:
                - architecture: Architecture analysis
                    - layer_mapping: Layer correspondence
                    - capacity: Capacity matching
                - transformation: Transformation analysis
                    - adaptation: Adaptation metrics
                    - alignment: Alignment scores
                - information: Information flow analysis
                    - bottlenecks: Information bottlenecks
                    - compression: Information compression
                - recommendations: Adaptation recommendations
        """

    def _analyze_loss_components(
        self,
        loss_values: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze different components of the distillation loss.
        
        Examines contribution and behavior of loss components:
        - Soft targets loss
        - Hard targets loss
        - Feature matching loss
        - Auxiliary losses
        
        Args:
            loss_values: Values of different loss components
            gradients: Gradients from different losses
            
        Returns:
            Dict containing:
                - contribution: Loss contribution analysis
                    - weights: Component weights
                    - importance: Component importance
                - gradient: Gradient analysis
                    - magnitude: Gradient magnitudes
                    - direction: Gradient directions
                - interaction: Loss interaction analysis
                    - correlation: Component correlations
                    - conflict: Gradient conflicts
                - dynamics: Loss dynamics analysis
                    - evolution: Loss evolution
                    - balance: Component balance
        """

    def _analyze_training_dynamics(
        self,
        train_history: List[Dict[str, float]],
        validation_history: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze training dynamics and convergence behavior.
        
        Examines training progression and stability through:
        - Loss trajectory analysis
        - Learning rate dynamics
        - Gradient behavior
        - Validation performance
        
        Args:
            train_history: Training metrics per iteration
            validation_history: Validation metrics per iteration
            
        Returns:
            Dict containing:
                - loss_dynamics: Loss behavior analysis
                    - convergence: Convergence patterns
                    - stability: Loss stability metrics
                - learning_rate: Learning rate analysis
                    - adaptation: LR adaptation patterns
                    - effectiveness: LR effectiveness metrics
                - gradient_behavior: Gradient analysis
                    - magnitude: Gradient magnitude trends
                    - direction: Gradient direction stability
                - validation: Validation performance analysis
                    - generalization: Generalization metrics
                    - overfitting: Overfitting indicators
        """

    def _analyze_memory_usage(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict[str, Any]:
        """Analyze memory usage patterns of models.
        
        Evaluates memory efficiency through:
        - Static memory requirements
        - Dynamic memory allocation
        - Memory scaling behavior
        - Memory access patterns
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dict containing:
                - static_memory: Static memory analysis
                    - parameters: Parameter memory usage
                    - buffers: Buffer memory usage
                - dynamic_memory: Dynamic memory analysis
                    - activation: Activation memory usage
                    - gradient: Gradient memory usage
                - scaling: Memory scaling analysis
                    - batch_scaling: Batch size scaling
                    - feature_scaling: Feature size scaling
                - efficiency: Memory efficiency metrics
                    - utilization: Memory utilization rates
                    - overhead: Memory overhead analysis
        """

    def _analyze_computational_complexity(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        input_shapes: List[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        """Analyze computational complexity of models.
        
        Evaluates computational efficiency through:
        - FLOPs analysis
        - Operation distribution
        - Layer-wise complexity
        - Throughput analysis
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            input_shapes: List of input shapes to test
            
        Returns:
            Dict containing:
                - flops: FLOPs analysis
                    - total: Total FLOPs count
                    - distribution: FLOPs distribution
                - operations: Operation analysis
                    - types: Operation type distribution
                    - costs: Operation-wise costs
                - layers: Layer-wise analysis
                    - complexity: Layer complexity metrics
                    - bottlenecks: Computational bottlenecks
                - throughput: Throughput analysis
                    - theoretical: Theoretical throughput
                    - measured: Measured throughput
        """

    def _analyze_model_architecture(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze model architecture characteristics.
        
        Examines architectural aspects:
        - Layer structure
        - Parameter distribution
        - Connectivity patterns
        - Architectural efficiency
        
        Args:
            student_model: Student model architecture
            teacher_model: Teacher model architecture
            
        Returns:
            Dict containing:
                - structure: Structural analysis
                    - layers: Layer hierarchy analysis
                    - connections: Connection patterns
                - parameters: Parameter analysis
                    - distribution: Parameter distribution
                    - sharing: Parameter sharing patterns
                - capacity: Capacity analysis
                    - layer_capacity: Layer-wise capacity
                    - bottlenecks: Capacity bottlenecks
                - efficiency: Architecture efficiency
                    - design: Design efficiency metrics
                    - utilization: Resource utilization
        """

    def _analyze_feature_distribution(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze distribution of feature representations.
        
        Examines feature characteristics:
        - Statistical properties
        - Distribution matching
        - Feature space coverage
        - Representation quality
        
        Args:
            student_features: Student model feature tensors
            teacher_features: Teacher model feature tensors
            
        Returns:
            Dict containing:
                - statistics: Statistical analysis
                    - moments: Distribution moments
                    - ranges: Value ranges
                - matching: Distribution matching
                    - kl_div: KL divergence
                    - wasserstein: Wasserstein distance
                - coverage: Feature space analysis
                    - density: Density estimation
                    - gaps: Coverage gaps
                - quality: Representation quality
                    - diversity: Feature diversity
                    - redundancy: Feature redundancy
        """

    def _analyze_gradient_flow(
        self,
        model: nn.Module,
        loss_fn: callable,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze gradient flow through the model.
        
        Examines gradient propagation:
        - Gradient magnitudes
        - Layer-wise gradients
        - Gradient path analysis
        - Vanishing/exploding gradients
        
        Args:
            model: Model to analyze gradients
            loss_fn: Loss function
            data: Input data batch
            labels: Ground truth labels
            
        Returns:
            Dict containing:
                - magnitudes: Gradient magnitude analysis
                    - statistics: Magnitude statistics
                    - distribution: Magnitude distribution
                - layer_gradients: Layer-wise analysis
                    - values: Layer gradient values
                    - ratios: Layer gradient ratios
                - paths: Gradient path analysis
                    - flow: Gradient flow patterns
                    - bottlenecks: Flow bottlenecks
                - issues: Gradient issues
                    - vanishing: Vanishing gradient detection
                    - exploding: Exploding gradient detection
        """

    def _compute_feature_statistics(
        self,
        features: torch.Tensor,
        layer_name: str = None
    ) -> Dict[str, Any]:
        """Compute statistical measures of feature activations.
        
        Calculates various statistical properties of feature maps to understand
        activation patterns and distributions.
        
        Args:
            features: Feature tensor (N x C x H x W)
            layer_name: Optional name of layer for logging
            
        Returns:
            Dict containing:
                - basic_stats: Basic statistics
                    - mean: Channel-wise means
                    - std: Channel-wise standard deviations
                    - min: Minimum values
                    - max: Maximum values
                - distribution: Distribution analysis
                    - skewness: Distribution skewness
                    - kurtosis: Distribution kurtosis
                - sparsity: Sparsity analysis
                    - zero_fraction: Fraction of zero activations
                    - effective_range: Range of non-zero values
        """

    def _compute_layer_correlations(
        self,
        student_layer: torch.Tensor,
        teacher_layer: torch.Tensor,
        correlation_type: str = 'pearson'
    ) -> Dict[str, float]:
        """Compute correlations between student and teacher layer features.
        
        Calculates different correlation metrics between corresponding layers
        to measure feature alignment.
        
        Args:
            student_layer: Student layer features (N x C x H x W)
            teacher_layer: Teacher layer features (N x C x H x W)
            correlation_type: Type of correlation to compute
                ('pearson', 'spearman', or 'cka')
            
        Returns:
            Dict containing:
                - correlation: Main correlation score
                - channel_correlations: Channel-wise correlations
                - spatial_correlations: Spatial correlations
                - significance: Statistical significance
        """

    def _compute_gradient_metrics(
        self,
        gradients: List[torch.Tensor],
        names: List[str] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive gradient analysis metrics.
        
        Analyzes gradient behavior through multiple metrics to understand
        optimization dynamics.
        
        Args:
            gradients: List of gradient tensors
            names: Optional list of parameter names
            
        Returns:
            Dict containing:
                - magnitude: Gradient magnitude analysis
                    - mean: Mean gradient magnitude
                    - std: Standard deviation
                    - norm: Gradient norm
                - direction: Gradient direction analysis
                    - cosine_similarity: Direction similarities
                    - angle: Angular statistics
                - distribution: Distribution analysis
                    - histogram: Gradient histogram
                    - percentiles: Key percentiles
        """

    def _compute_stability_score(
        self,
        metrics: Dict[str, List[float]],
        window_size: int = 10
    ) -> Dict[str, float]:
        """Compute training stability scores from metrics.
        
        Analyzes metric trajectories to assess training stability
        and convergence quality.
        
        Args:
            metrics: Dictionary of metric histories
            window_size: Window size for moving statistics
            
        Returns:
            Dict containing:
                - overall_stability: Overall stability score
                - metric_stability: Per-metric stability scores
                - trend_analysis: Trend analysis results
                - oscillation_scores: Oscillation detection scores
        """

    def _compute_efficiency_metrics(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Compute model efficiency metrics.
        
        Evaluates computational and memory efficiency through
        various performance metrics.
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            batch_size: Batch size for analysis
            
        Returns:
            Dict containing:
                - compute_efficiency: Computational efficiency scores
                - memory_efficiency: Memory usage efficiency
                - throughput: Processing throughput metrics
                - latency: Processing latency metrics
        """

    def _compute_robustness_scores(
        self,
        predictions: List[torch.Tensor],
        perturbations: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute model robustness scores.
        
        Evaluates prediction stability under various perturbations
        and input transformations.
        
        Args:
            predictions: List of model predictions
            perturbations: List of input perturbations
            
        Returns:
            Dict containing:
                - stability: Prediction stability scores
                - sensitivity: Input sensitivity measures
                - consistency: Prediction consistency scores
                - invariance: Feature invariance metrics
        """

    def _compute_adaptation_metrics(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute layer adaptation quality metrics.
        
        Evaluates how well student layers adapt to match
        teacher layer representations.
        
        Args:
            student_features: List of student layer features
            teacher_features: List of teacher layer features
            
        Returns:
            Dict containing:
                - alignment: Feature alignment scores
                - adaptation: Adaptation quality metrics
                - matching: Feature matching scores
                - transfer: Knowledge transfer metrics
        """

    def _compute_optimization_quality(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute optimization quality metrics.
        
        Analyzes training and validation metrics to assess
        optimization effectiveness.
        
        Args:
            train_metrics: Training metric histories
            val_metrics: Validation metric histories
            
        Returns:
            Dict containing:
                - convergence: Convergence quality scores
                - generalization: Generalization metrics
                - efficiency: Optimization efficiency
                - stability: Training stability scores
        """

    def _analyze_model_capacity(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze model capacity and representation power.
        
        Examines model capacity characteristics:
        - Parameter efficiency
        - Representation power
        - Layer capacity
        - Information bottlenecks
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            
        Returns:
            Dict containing:
                - parameters: Parameter analysis
                    - efficiency: Parameter efficiency
                    - distribution: Parameter distribution
                - representation: Representation analysis
                    - power: Representation power
                    - complexity: Model complexity
                - layers: Layer-wise analysis
                    - capacity: Layer capacity
                    - bottlenecks: Capacity bottlenecks
                - compression: Compression analysis
                    - ratio: Compression ratio
                    - quality: Compression quality
        """

    def _analyze_generalization_capability(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze model generalization capability.
        
        Evaluates how well models generalize:
        - Train-test gap
        - Out-of-distribution behavior
        - Robustness
        - Uncertainty estimation
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            train_loader: Training data loader
            test_loader: Test data loader
            device: Computation device
            
        Returns:
            Dict containing:
                - performance: Performance analysis
                    - train: Training performance
                    - test: Test performance
                - robustness: Robustness analysis
                    - noise: Noise robustness
                    - perturbation: Perturbation robustness
                - uncertainty: Uncertainty analysis
                    - calibration: Calibration metrics
                    - reliability: Reliability scores
                - distribution: Distribution analysis
                    - shift: Distribution shift impact
                    - adaptation: Adaptation capability
        """

    def _analyze_feature_redundancy(
        self,
        features: List[torch.Tensor],
        threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Analyze feature redundancy and information content.
        
        Examines feature characteristics:
        - Feature correlations
        - Information overlap
        - Redundant dimensions
        - Essential features
        
        Args:
            features: List of feature tensors to analyze
            threshold: Correlation threshold for redundancy
            
        Returns:
            Dict containing:
                - correlations: Correlation analysis
                    - matrix: Correlation matrix
                    - clusters: Feature clusters
                - redundancy: Redundancy analysis
                    - ratio: Redundancy ratio
                    - groups: Redundant groups
                - information: Information analysis
                    - content: Information content
                    - uniqueness: Feature uniqueness
                - essential: Essential feature analysis
                    - features: Essential features
                    - importance: Feature importance
        """

    def _analyze_optimization_stability(
        self,
        loss_history: List[float],
        grad_history: List[torch.Tensor],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze optimization stability and convergence.
        
        Examines optimization characteristics:
        - Loss landscape
        - Gradient behavior
        - Convergence stability
        - Learning dynamics
        
        Args:
            loss_history: History of loss values
            grad_history: History of gradients
            window_size: Window for moving statistics
            
        Returns:
            Dict containing:
                - loss: Loss analysis
                    - trajectory: Loss trajectory
                    - stability: Loss stability
                - gradients: Gradient analysis
                    - magnitude: Gradient magnitudes
                    - direction: Gradient directions
                - convergence: Convergence analysis
                    - speed: Convergence speed
                    - quality: Convergence quality
                - dynamics: Learning dynamics
                    - oscillation: Oscillation patterns
                    - adaptation: Adaptation behavior
        """

    def _analyze_model_similarity(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze structural and behavioral similarity between models.
        
        Examines model similarities:
        - Architecture similarity
        - Parameter distributions
        - Feature representations
        - Decision patterns
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            
        Returns:
            Dict containing:
                - structure: Structural similarity
                    - architecture: Architecture similarity
                    - parameters: Parameter similarity
                - behavior: Behavioral similarity
                    - features: Feature similarity
                    - decisions: Decision similarity
                - alignment: Model alignment
                    - layers: Layer alignment
                    - functions: Functional alignment
                - differences: Key differences
                    - structural: Structural differences
                    - functional: Functional differences
        """

    def _analyze_compression_efficiency(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze efficiency of model compression.
        
        Evaluates compression effectiveness:
        - Size reduction
        - Performance retention
        - Resource efficiency
        - Quality tradeoffs
        
        Args:
            student_model: Compressed student model
            teacher_model: Original teacher model
            metrics: Performance metrics
            
        Returns:
            Dict containing:
                - reduction: Size reduction analysis
                    - parameters: Parameter reduction
                    - computation: Computation reduction
                - retention: Performance retention
                    - accuracy: Accuracy retention
                    - quality: Quality retention
                - efficiency: Resource efficiency
                    - memory: Memory efficiency
                    - compute: Compute efficiency
                - tradeoffs: Quality tradeoffs
                    - size_quality: Size vs quality
                    - speed_accuracy: Speed vs accuracy
        """

    def _analyze_prediction_patterns(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze prediction patterns and behaviors.
        
        Examines prediction characteristics and relationships:
        - Agreement patterns
        - Error correlations
        - Decision boundaries
        - Confidence patterns
        
        Args:
            student_preds: Student model predictions (N x C)
            teacher_preds: Teacher model predictions (N x C)
            true_labels: Ground truth labels (N)
            
        Returns:
            Dict containing:
                - agreement: Agreement analysis
                    - overall: Overall agreement rate
                    - class_wise: Class-wise agreement
                - errors: Error analysis
                    - correlation: Error correlation patterns
                    - distribution: Error distribution
                - decisions: Decision analysis
                    - boundaries: Decision boundary analysis
                    - regions: Decision region overlap
                - confidence: Confidence analysis
                    - calibration: Confidence calibration
                    - distribution: Confidence distribution
        """

    def _analyze_feature_evolution(
        self,
        feature_history: List[Dict[str, torch.Tensor]],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze evolution of feature representations during training.
        
        Tracks changes in feature characteristics:
        - Representation stability
        - Feature adaptation
        - Information flow
        - Convergence patterns
        
        Args:
            feature_history: History of feature representations
            window_size: Window for moving statistics
            
        Returns:
            Dict containing:
                - stability: Stability analysis
                    - temporal: Temporal stability metrics
                    - spatial: Spatial stability metrics
                - adaptation: Adaptation analysis
                    - speed: Adaptation speed
                    - quality: Adaptation quality
                - information: Information flow
                    - retention: Information retention
                    - transfer: Information transfer
                - convergence: Convergence analysis
                    - trajectory: Feature trajectories
                    - patterns: Convergence patterns
        """

    def _analyze_layer_interactions(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze interactions between model layers.
        
        Examines layer relationships and dependencies:
        - Information flow
        - Feature transformations
        - Skip connections
        - Layer coupling
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            data: Input data batch
            
        Returns:
            Dict containing:
                - flow: Information flow analysis
                    - forward: Forward propagation patterns
                    - backward: Backward propagation patterns
                - transformations: Feature transformation analysis
                    - mappings: Layer-wise mappings
                    - complexity: Transformation complexity
                - connections: Connection analysis
                    - direct: Direct connections
                    - skip: Skip connection patterns
                - coupling: Layer coupling analysis
                    - strength: Coupling strength
                    - dependencies: Layer dependencies
        """

    def _analyze_training_efficiency(
        self,
        train_history: Dict[str, List[float]],
        resource_usage: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze training efficiency and resource utilization.
        
        Evaluates training process efficiency:
        - Convergence speed
        - Resource utilization
        - Training stability
        - Cost effectiveness
        
        Args:
            train_history: Training metric history
            resource_usage: Resource usage metrics
            
        Returns:
            Dict containing:
                - convergence: Convergence analysis
                    - speed: Convergence speed metrics
                    - quality: Convergence quality
                - resources: Resource analysis
                    - utilization: Resource utilization
                    - efficiency: Resource efficiency
                - stability: Stability analysis
                    - metrics: Stability metrics
                    - issues: Stability issues
                - cost: Cost analysis
                    - computation: Computational cost
                    - memory: Memory cost
        """

    def _analyze_model_behavior(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze model behavior and characteristics.
        
        Examines model behavior patterns:
        - Prediction patterns
        - Feature extraction
        - Decision making
        - Robustness characteristics
        
        Args:
            student_model: Student model to analyze
            teacher_model: Teacher model for comparison
            data_loader: Data loader for evaluation
            device: Computation device
            
        Returns:
            Dict containing:
                - predictions: Prediction analysis
                    - patterns: Prediction patterns
                    - consistency: Prediction consistency
                - features: Feature analysis
                    - extraction: Feature extraction patterns
                    - hierarchy: Feature hierarchy
                - decisions: Decision analysis
                    - making: Decision making process
                    - boundaries: Decision boundaries
                - robustness: Robustness analysis
                    - stability: Behavioral stability
                    - adaptability: Adaptation capability
        """

    def _compute_confusion_matrix(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        """Compute confusion matrix from predictions.
        
        Calculates confusion matrix to analyze prediction patterns
        and error distributions.
        
        Args:
            predictions: Model predictions (N)
            labels: Ground truth labels (N)
            
        Returns:
            Numpy array containing confusion matrix
        """

    def _compute_class_metrics(
        self,
        confusion_matrix: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-class performance metrics.
        
        Calculates precision, recall, and F1-score for each class
        from confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix from predictions
            
        Returns:
            Dict mapping class indices to metrics:
                - precision: Class precision
                - recall: Class recall
                - f1: F1 score
    """

    def _compute_calibration_metrics(
        self,
        confidences: torch.Tensor,
        accuracies: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute calibration metrics for model predictions.
        
        Evaluates how well model confidence aligns with accuracy:
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Reliability diagram data
        
        Args:
            confidences: Prediction confidence scores (N)
            accuracies: Binary accuracy indicators (N)
            n_bins: Number of bins for reliability diagram
            
        Returns:
            Dict containing:
                - ece: Expected Calibration Error
                - mce: Maximum Calibration Error
                - reliability: Reliability diagram data
                    - confidences: Mean confidence per bin
                    - accuracies: Mean accuracy per bin
                    - counts: Samples per bin
        """

    def _compute_feature_similarity(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        similarity_type: str = 'cosine'
    ) -> Dict[str, torch.Tensor]:
        """Compute similarity between feature representations.
        
        Calculates various similarity metrics between feature maps:
        - Cosine similarity
        - L2 distance
        - Correlation coefficient
        - CKA similarity
        
        Args:
            student_features: Student model features (N x C x H x W)
            teacher_features: Teacher model features (N x C x H x W)
            similarity_type: Type of similarity metric to use
            
        Returns:
            Dict containing:
                - similarity: Main similarity score
                - channel_sim: Channel-wise similarities
                - spatial_sim: Spatial similarities
                - stats: Similarity statistics
    """

    def _compute_gradient_diversity(
        self,
        gradients: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute diversity metrics for gradients.
        
        Analyzes gradient diversity to understand optimization behavior:
        - Direction diversity
        - Magnitude diversity
        - Principal components
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Dict containing:
                - direction_diversity: Gradient direction diversity
                - magnitude_diversity: Gradient magnitude diversity
                - principal_angles: Principal angles between gradients
                - effective_rank: Effective rank of gradient space
    """

    def _compute_loss_surface_metrics(
        self,
        loss_values: List[float],
        parameters: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics describing loss surface.
        
        Analyzes loss surface characteristics:
        - Smoothness
        - Curvature
        - Critical points
        - Basin structure
        
        Args:
            loss_values: Loss values at sampled points
            parameters: Model parameters at sampled points
            
        Returns:
            Dict containing:
                - smoothness: Surface smoothness metrics
                - curvature: Local curvature metrics
                - critical_points: Critical point analysis
                - basin_metrics: Loss basin characteristics
    """

    def _plot_training_dynamics(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]]
    ) -> go.Figure:
        """Plot training dynamics analysis.
        
        Creates visualization of training progress including:
        - Learning curves
        - Loss trajectories
        - Metric evolution
        - Validation performance
        
        Args:
            train_history: Training metrics history
            val_history: Validation metrics history
            
        Returns:
            Plotly figure containing:
                - Loss plots: Training and validation loss curves
                - Metric plots: Performance metric trajectories
                - Gradient plots: Gradient statistics over time
                - Analysis plots: Additional training analysis
        """

    def _plot_feature_analysis(
        self,
        feature_analysis: Dict[str, Any],
        max_features: int = 16
    ) -> go.Figure:
        """Plot feature analysis results.
        
        Creates comprehensive visualization of feature characteristics:
        - Feature maps
        - Activation patterns
        - Feature correlations
        - Layer comparisons
        
        Args:
            feature_analysis: Feature analysis results
            max_features: Maximum number of features to plot
            
        Returns:
            Plotly figure containing:
                - Feature maps: Spatial feature visualizations
                - Activation plots: Activation pattern analysis
                - Correlation plots: Feature correlation matrices
                - Comparison plots: Student-teacher comparisons
        """

    def _plot_model_comparison(
        self,
        student_metrics: Dict[str, float],
        teacher_metrics: Dict[str, float]
    ) -> go.Figure:
        """Plot comparative analysis of student and teacher models.
        
        Creates side-by-side comparison visualizations:
        - Performance metrics
        - Resource usage
        - Behavioral patterns
        - Quality metrics
        
        Args:
            student_metrics: Student model metrics
            teacher_metrics: Teacher model metrics
            
        Returns:
            Plotly figure containing:
                - Performance plots: Performance comparison
                - Resource plots: Resource usage comparison
                - Behavior plots: Behavioral pattern comparison
                - Quality plots: Quality metric comparison
        """

    def _plot_optimization_analysis(
        self,
        optimization_metrics: Dict[str, Any]
    ) -> go.Figure:
        """Plot optimization analysis results.
        
        Visualizes optimization behavior and characteristics:
        - Convergence patterns
        - Gradient behavior
        - Loss landscape
        - Training stability
        
        Args:
            optimization_metrics: Optimization analysis results
            
        Returns:
            Plotly figure containing:
                - Convergence plots: Convergence visualization
                - Gradient plots: Gradient analysis plots
                - Landscape plots: Loss landscape visualization
                - Stability plots: Training stability analysis
        """

    def generate_quality_report(
        self,
        quality_results: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Generate comprehensive quality analysis report.
        
        Creates detailed report including:
        - Quality metrics summary
        - Performance analysis
        - Training dynamics
        - Optimization analysis
        - Recommendations
        
        Args:
            quality_results: Quality analysis results
            output_dir: Directory to save report
            
        Returns:
            None. Saves report files to specified directory:
                - HTML report with interactive visualizations
                - JSON data file with analysis results
                - PDF summary report
                - Visualization files
        """

    def _evaluate_model(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model performance comprehensively.
        
        Computes various performance metrics:
        - Accuracy metrics
        - Loss metrics
        - Prediction quality
        - Resource usage
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Computation device
            
        Returns:
            Dict containing:
                - accuracy: Accuracy metrics
                    - top1: Top-1 accuracy
                    - top5: Top-5 accuracy
                - loss: Loss metrics
                    - total: Total loss
                    - components: Loss components
                - predictions: Prediction metrics
                    - confidence: Confidence scores
                    - calibration: Calibration metrics
                - resources: Resource metrics
                    - memory: Memory usage
                    - compute: Computation time
        """

    def _compute_quality_metrics(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive quality metrics.
        
        Evaluates multiple aspects of model quality:
        - Performance quality
        - Prediction quality
        - Feature quality
        - Efficiency quality
        
        Args:
            student_model: Student model to evaluate
            teacher_model: Teacher model for comparison
            data: Input data batch
            labels: Ground truth labels
            
        Returns:
            Dict containing:
                - performance: Performance quality metrics
                - predictions: Prediction quality metrics
                - features: Feature quality metrics
                - efficiency: Efficiency quality metrics
        """

    def _plot_training_dynamics(
        self,
        dynamics: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Plot training dynamics analysis results.
        
        Args:
            dynamics: Training dynamics data from analyze_training_dynamics()
            save_path: Optional path to save plots
            
        Returns:
            Dictionary containing:
                - figures: Generated plot figures
                - metrics: Computed dynamics metrics
        """
        # ... existing implementation ...

    def analyze_feature_quality(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze quality of learned feature representations.
        
        Args:
            student_features: Feature vectors from student model
            teacher_features: Feature vectors from teacher model
            
        Returns:
            Dictionary of feature quality metrics:
                - cosine_similarity: Cosine similarity between feature spaces
                - feature_collapse: Feature collapse metric
                - feature_diversity: Feature diversity score
        """
        # ... existing implementation ...

    def analyze_prediction_quality(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze quality of model predictions.
        
        Args:
            student_preds: Predictions from student model
            teacher_preds: Predictions from teacher model
            labels: Ground truth labels
            
        Returns:
            Dictionary of prediction quality metrics:
                - accuracy: Prediction accuracy
                - calibration: Prediction calibration score
                - confidence: Mean prediction confidence
        """
        # ... existing implementation ...