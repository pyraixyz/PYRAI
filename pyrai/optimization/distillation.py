"""Knowledge distillation module for model compression.

This module implements knowledge distillation techniques to transfer knowledge from a large teacher model 
to a smaller student model. It supports various distillation methods and loss functions.

Example:
    ```python
    # Create distillation config
    config = DistillationConfig(
        temperature=4.0,
        alpha=0.5,
        method=DistillationMethod.SOFT_TARGETS
    )

    # Initialize distillation
    distiller = KnowledgeDistillation(
        teacher_model=teacher,
        student_model=student,
        config=config
    )

    # Train with distillation
    distiller.train(train_loader, epochs=10)
    ```
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from collections import defaultdict
import copy

try:
    import matplotlib.pyplot as plt # type: ignore
except ImportError:
    plt = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .optimizer import ModelOptimizer
from .distillation_monitor import DistillationMetrics, DistillationMonitor
from .distillation_tuner import DistillationTuner, TuningConfig, TuningMode
from .distillation_visualizer import DistillationVisualizer
from .metrics_collector import MetricsCollector

class DistillationMethod(Enum):
    """Distillation methods."""
    SOFT_TARGETS = "soft_targets"  # Standard soft targets
    FEATURE_MATCHING = "feature_matching"  # Match intermediate features
    ATTENTION = "attention"  # Attention transfer
    RELATION = "relation"  # Relational knowledge distillation

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation.
    
    Args:
        temperature: Temperature parameter for softening probability distributions
        alpha: Weight balancing cross-entropy and distillation losses
        method: Distillation method to use
        feature_layers: Layers to use for feature matching (if applicable)
        attention_layers: Layers to use for attention transfer (if applicable)
    """
    temperature: float = 4.0
    alpha: float = 0.5
    method: DistillationMethod = DistillationMethod.SOFT_TARGETS
    feature_layers: Optional[List[str]] = None
    attention_layers: Optional[List[str]] = None

class DistillationOptimizer:
    """Optimizes model compression through knowledge distillation.
    
    This class implements advanced knowledge distillation techniques to compress
    large teacher models into smaller student models while maintaining performance.
    Supports multiple distillation methods and loss functions.
    
    Features:
    - Multiple distillation methods (soft targets, feature matching, attention)
    - Adaptive temperature scheduling
    - Multi-teacher distillation
    - Progressive knowledge transfer
    - Loss function customization
    
    Example:
        ```python
        optimizer = DistillationOptimizer(
            teacher_model=teacher,
            config=DistillationConfig(
                temperature=4.0,
                alpha=0.5,
                method=DistillationMethod.SOFT_TARGETS
            )
        )
        
        # Train student model
        student = optimizer.distill(
            student_model=student,
            train_data=train_loader,
            epochs=10
        )
        ```
    """
    
    def __init__(
        self,
        teacher_model: Union["torch.nn.Module", "tf.keras.Model"], # type: ignore
        config: "DistillationConfig"
    ):
        """Initialize distillation optimizer.
        
        Args:
            teacher_model: Teacher model
            config: Distillation configuration
        """
        self.teacher_model = teacher_model
        self.config = config
        self.is_torch = isinstance(teacher_model, torch.nn.Module)
        
    def distill(
        self,
        student_model: Union["torch.nn.Module", "tf.keras.Model"], # type: ignore
        train_data: Any,
        val_data: Optional[Any] = None,
        **kwargs
    ) -> Union["torch.nn.Module", "tf.keras.Model"]: # type: ignore
        """Perform knowledge distillation.
        
        Args:
            student_model: Student model to train
            train_data: Training data
            val_data: Optional validation data
            **kwargs: Additional training arguments
            
        Returns:
            Trained student model
        """
        if self.is_torch:
            return self._distill_torch(student_model, train_data, val_data, **kwargs)
        else:
            return self._distill_tf(student_model, train_data, val_data, **kwargs)
            
    def _distill_torch(
        self,
        student_model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader] = None,
        **kwargs
    ) -> torch.nn.Module:
        """PyTorch distillation implementation."""
        device = next(student_model.parameters()).device
        self.teacher_model = self.teacher_model.to(device)
        self.teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(), **kwargs.get("optimizer_config", {}))
        
        for epoch in range(kwargs.get("epochs", 10)):
            student_model.train()
            for batch_idx, (data, target) in enumerate(train_data):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                # Get student predictions
                student_logits = student_model(data)
                
                # Calculate loss based on method
                if self.config.method == DistillationMethod.SOFT_TARGETS:
                    loss = self._compute_soft_targets_loss(
                        student_logits,
                        teacher_logits,
                        target
                    )
                elif self.config.method == DistillationMethod.ATTENTION:
                    loss = self._attention_distillation_loss(
                        student_model,
                        self.teacher_model,
                        data,
                        target
                    )
                elif self.config.method == DistillationMethod.FEATURE_MATCHING:
                    loss = self._feature_distillation_loss(
                        student_model,
                        self.teacher_model,
                        data,
                        target
                    )
                else:  # RELATION
                    loss = self._relation_distillation_loss(
                        student_logits,
                        teacher_logits
                    )
                    
                loss.backward()
                optimizer.step()
                
            # Validation
            if val_data:
                student_model.eval()
                val_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in val_data:
                        data, target = data.to(device), target.to(device)
                        output = student_model(data)
                        val_loss += F.cross_entropy(output, target).item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        
                print(f"Epoch {epoch}: Validation Accuracy = {100. * correct / len(val_data.dataset):.2f}%")
                
        return student_model
        
    def _compute_soft_targets_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss with soft targets.
        
        Combines cross-entropy loss on hard targets with KL divergence loss
        between softened teacher and student predictions.
        
        The temperature parameter controls prediction softening:
        - Higher temperature → Softer predictions
        - Lower temperature → Sharper predictions
        
        Args:
            student_logits: Raw student model outputs
            teacher_logits: Raw teacher model outputs
            target: Ground truth labels
            
        Returns:
            Combined distillation loss:
                alpha * soft_targets_loss + (1-alpha) * hard_targets_loss
                
        Example:
            ```python
            loss = optimizer._compute_soft_targets_loss(
                student_out, teacher_out, labels
            )
            loss.backward()
            ```
        """
        T = self.config.temperature
        alpha = self.config.alpha
        
        # Soft targets loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, target)
        
        return alpha * soft_loss + (1 - alpha) * hard_loss
        
    def _attention_distillation_loss(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Attention transfer distillation loss."""
        def get_attention_maps(model, input_tensor):
            attention_maps = []
            for module in model.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    # Get attention weights
                    _, weights = module(input_tensor, input_tensor, input_tensor)
                    attention_maps.append(weights)
            return attention_maps
            
        student_attention = get_attention_maps(student_model, data)
        teacher_attention = get_attention_maps(teacher_model, data)
        
        attention_loss = 0
        for s_att, t_att in zip(student_attention, teacher_attention):
            attention_loss += F.mse_loss(s_att, t_att)
            
        return self._compute_soft_targets_loss(
            student_model(data),
            self.teacher_model(data),
            target
        ) + attention_loss
        
    def _feature_distillation_loss(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Feature matching distillation loss."""
        def get_features(model, input_tensor):
            features = []
            for name, module in model.named_modules():
                if name in (self.config.feature_layers or []):
                    features.append(module(input_tensor))
            return features
            
        student_features = get_features(student_model, data)
        teacher_features = get_features(teacher_model, data)
        
        feature_loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Adapt feature dimensions if needed
            if s_feat.shape != t_feat.shape:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
            feature_loss += F.mse_loss(s_feat, t_feat)
            
        return self._compute_soft_targets_loss(
            student_model(data),
            self.teacher_model(data),
            target
        ) + feature_loss
        
    def _relation_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute relational knowledge distillation loss.
        
        Preserves structural relationships between instances by matching
        pairwise distance matrices between teacher and student outputs.
        
        Args:
            student_logits: Student model output logits
                Shape: [batch_size, num_classes]
            teacher_logits: Teacher model output logits
                Shape: [batch_size, num_classes]
                
        Returns:
            KL divergence loss between distance probability distributions
            
        Example:
            ```python
            loss = distiller._relation_distillation_loss(
                student_out["logits"],
                teacher_out["logits"]
            )
            ```
        """
        # Compute distance matrices
        s_dist = self._compute_distance_matrix_torch(student_logits)
        t_dist = self._compute_distance_matrix_torch(teacher_logits)
        
        # Convert to probability distributions
        s_prob = torch.softmax(s_dist / self.config.temperature, dim=-1)
        t_prob = torch.softmax(t_dist / self.config.temperature, dim=-1)
        
        # KL divergence loss
        loss = torch.nn.KLDivLoss(reduction='batchmean')(
            torch.log(s_prob),
            t_prob
        ) * (self.config.temperature ** 2)
        
        return loss
        
    def _compute_distance_matrix_torch(
        self,
        features: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute pairwise distance/similarity matrix for PyTorch tensors.
        
        Computes either Euclidean distances or cosine similarities between
        all pairs of feature vectors in the batch.
        
        Args:
            features: Input feature tensor
                Shape: [batch_size, num_instances, feature_dim]
                
        Returns:
            Distance/similarity matrix
                Shape: [batch_size, num_instances, num_instances]
                
        Example:
            ```python
            # Compute pairwise distances
            dist_mat = distiller._compute_distance_matrix_torch(features)
            
            # Convert to probabilities
            probs = torch.softmax(dist_mat / temperature, dim=-1)
            ```
        """
        if self.distance_type == "euclidean":
            # [B, N, D] -> [B, N, N]
            dist = torch.cdist(features, features)
        else:  # cosine
            # Normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            # Compute cosine similarity
            dist = torch.matmul(features, features.transpose(-2, -1))
            
        return dist

class DistillationLoss:
    """Knowledge distillation loss functions.
    
    Collection of loss functions used in knowledge distillation:
    - KL divergence loss for soft targets
    - Attention transfer loss
    - Feature matching loss
    - Mutual KL divergence for multi-teacher distillation
    
    Example:
        ```python
        # Compute soft targets loss
        loss = DistillationLoss.kl_div(
            student_logits=student_out,
            teacher_logits=teacher_out,
            temperature=4.0
        )
        ```
    """
    
    @staticmethod
    def kl_div(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """KL divergence loss."""
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature * temperature)
        
    @staticmethod
    def attention_transfer(
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """Attention transfer loss."""
        # Normalize attention maps
        student_attention = student_attention.view(student_attention.size(0), -1)
        teacher_attention = teacher_attention.view(teacher_attention.size(0), -1)
        
        student_attention = F.normalize(student_attention, p=2, dim=-1)
        teacher_attention = F.normalize(teacher_attention, p=2, dim=-1)
        
        return F.mse_loss(student_attention, teacher_attention)
        
    @staticmethod
    def feature_matching(
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Feature matching loss between teacher and student.
        
        Matches the statistics (mean and standard deviation) of feature maps
        between teacher and student models. This helps transfer the feature
        representation knowledge.
        
        Args:
            student_features: Feature maps from student model
                Shape: [batch_size, channels, height, width]
            teacher_features: Feature maps from teacher model
                Shape: [batch_size, channels, height, width]
                
        Returns:
            Combined loss of mean and standard deviation matching
            
        Example:
            ```python
            s_feat = student.get_features(x)
            t_feat = teacher.get_features(x)
            feat_loss = DistillationLoss.feature_matching(s_feat, t_feat)
            ```
        """
        # Match feature statistics
        student_mean = student_features.mean(dim=[2, 3])
        teacher_mean = teacher_features.mean(dim=[2, 3])
        
        student_std = student_features.std(dim=[2, 3])
        teacher_std = teacher_features.std(dim=[2, 3])
        
        mean_loss = F.mse_loss(student_mean, teacher_mean)
        std_loss = F.mse_loss(student_std, teacher_std)
        
        return mean_loss + std_loss
        
    @staticmethod
    def mutual_kl(
        logits_list: List[torch.Tensor],
        temperature: float
    ) -> torch.Tensor:
        """Mutual KL divergence loss for multi-teacher distillation.
        
        Computes pairwise KL divergence between all teacher-student pairs
        and averages them. This is useful when distilling knowledge from
        multiple teacher models.
        
        Args:
            logits_list: List of logits from all models (teachers + student)
                Each tensor shape: [batch_size, num_classes]
            temperature: Temperature for softening predictions
            
        Returns:
            Average pairwise KL divergence loss
            
        Example:
            ```python
            loss = DistillationLoss.mutual_kl(
                logits_list=[teacher1_out, teacher2_out, student_out],
                temperature=4.0
            )
            ```
        """
        n = len(logits_list)
        loss = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                loss += DistillationLoss.kl_div(
                    logits_list[i],
                    logits_list[j],
                    temperature
                )
                
        return loss / (n * (n - 1) / 2)

class FeatureDistillation:
    """Feature-based knowledge distillation implementation.
    
    This class implements feature-level knowledge distillation by matching
    intermediate feature representations between teacher and student models.
    Features:
    - Multi-layer feature matching
    - L2 normalized feature alignment
    - Framework-agnostic implementation
    - Weighted feature matching
    
    Example:
        ```python
        distiller = FeatureDistillation(alpha=0.5)
        
        # Get features from both models
        s_feats = student.get_features(x)
        t_feats = teacher.get_features(x)
        
        # Compute feature distillation loss
        loss = distiller(s_feats, t_feats)
        ```
    """
    
    def __init__(self, alpha: float = 0.5):
        """Initialize feature distillation.
        
        Args:
            alpha: Weight for feature distillation loss
        """
        self.alpha = alpha
        
    def __call__(
        self,
        student_features: List[Union["torch.Tensor", "tf.Tensor"]],  # type: ignore
        teacher_features: List[Union["torch.Tensor", "tf.Tensor"]]   # type: ignore
    ) -> Union["torch.Tensor", "tf.Tensor"]:  # type: ignore
        """Calculate feature distillation loss."""
        if isinstance(student_features[0], torch.Tensor):
            return self._torch_loss(student_features, teacher_features)
        else:
            return self._tf_loss(student_features, teacher_features)
            
    def _torch_loss(
        self,
        student_features: List["torch.Tensor"],
        teacher_features: List["torch.Tensor"]
    ) -> "torch.Tensor":
        """Compute PyTorch feature distillation loss."""
        total_loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_feat = torch.nn.functional.normalize(s_feat, dim=1)
            t_feat = torch.nn.functional.normalize(t_feat, dim=1)
            loss = torch.nn.MSELoss()(s_feat, t_feat)
            total_loss += loss
        return total_loss * self.alpha

    def _tf_loss(
        self,
        student_features: List["tf.Tensor"],  # type: ignore
        teacher_features: List["tf.Tensor"]   # type: ignore
    ) -> "tf.Tensor":  # type: ignore
        """Compute TensorFlow feature distillation loss."""
        total_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_feat = tf.nn.l2_normalize(s_feat, axis=1)
            t_feat = tf.nn.l2_normalize(t_feat, axis=1)
            loss = tf.reduce_mean(tf.square(s_feat - t_feat))
            total_loss += loss
        return total_loss * self.alpha

class AttentionDistillation:
    """Attention-based knowledge distillation implementation.
    
    This class implements attention transfer between teacher and student models
    by matching their attention maps. Features:
    - Multi-head attention transfer
    - Temperature-scaled attention matching
    - Cross-framework support (PyTorch/TensorFlow)
    - Attention map normalization
    
    Example:
        ```python
        distiller = AttentionDistillation(temperature=4.0)
        
        # Get attention maps
        s_att = student.get_attention_maps(x)
        t_att = teacher.get_attention_maps(x)
        
        # Compute attention transfer loss
        loss = distiller(s_att, t_att)
        ```
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize attention distillation.
        
        Args:
            temperature: Temperature for attention softmax
        """
        self.temperature = temperature
        
    def __call__(
        self,
        student_attention: Union["torch.Tensor", "tf.Tensor"],  # type: ignore
        teacher_attention: Union["torch.Tensor", "tf.Tensor"]   # type: ignore
    ) -> Union["torch.Tensor", "tf.Tensor"]:  # type: ignore
        """Calculate attention distillation loss."""
        if isinstance(student_attention, torch.Tensor):
            return self._torch_loss(student_attention, teacher_attention)
        else:
            return self._tf_loss(student_attention, teacher_attention)
            
    def _torch_loss(
        self,
        student_attention: "torch.Tensor",
        teacher_attention: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute PyTorch attention distillation loss.
        
        Matches attention distributions between teacher and student using
        temperature-scaled KL divergence loss.
        
        Args:
            student_attention: Student model attention maps
                Shape: [batch_size, num_heads, seq_len, seq_len]
            teacher_attention: Teacher model attention maps
                Shape: [batch_size, num_heads, seq_len, seq_len]
                
        Returns:
            KL divergence loss between normalized attention maps
            
        Example:
            ```python
            loss = distiller._torch_loss(
                student_attention=s_att,
                teacher_attention=t_att
            )
            loss.backward()
            ```
        """
        # Normalize attention maps
        s_att = torch.softmax(student_attention / self.temperature, dim=-1)
        t_att = torch.softmax(teacher_attention / self.temperature, dim=-1)
        
        # KL divergence loss
        loss = torch.nn.KLDivLoss(reduction='batchmean')(
            torch.log(s_att),
            t_att
        ) * (self.temperature ** 2)
        
        return loss
        
    def _tf_loss(
        self,
        student_attention: "tf.Tensor",  # type: ignore
        teacher_attention: "tf.Tensor"   # type: ignore
    ) -> "tf.Tensor":  # type: ignore
        """Compute TensorFlow attention distillation loss.
        
        TensorFlow implementation of attention transfer using temperature-scaled
        KL divergence between attention distributions.
        
        Args:
            student_attention: Student model attention maps
                Shape: [batch_size, num_heads, seq_len, seq_len]
            teacher_attention: Teacher model attention maps
                Shape: [batch_size, num_heads, seq_len, seq_len]
                
        Returns:
            KL divergence loss between normalized attention maps
            
        Example:
            ```python
            loss = distiller._tf_loss(
                student_attention=s_att,
                teacher_attention=t_att
            )
            ```
        """
        # Normalize attention maps
        s_att = tf.nn.softmax(student_attention / self.temperature)
        t_att = tf.nn.softmax(teacher_attention / self.temperature)
        
        # KL divergence loss
        loss = tf.keras.losses.KLDivergence()(
            t_att,
            s_att
        ) * (self.temperature ** 2)
        
        return loss

class IntermediateDistillation:
    """Intermediate layer knowledge distillation.
    
    This class implements distillation from intermediate layers of teacher model
    to student model. Features:
    - Layer-wise feature matching
    - Attention map transfer
    - Channel importance weighting
    - Adaptive feature transformation
    
    Example:
        ```python
        distiller = IntermediateDistillation(
            teacher_layers=["layer1", "layer2", "layer3"],
            student_layers=["block1", "block2", "block3"],
            weights=[0.3, 0.3, 0.4]
        )
        
        loss = distiller(
            student_features=student_feats,
            teacher_features=teacher_feats
        )
        ```
    """

    def __init__(
        self,
        layer_mapping: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0
    ):
        """Initialize intermediate distillation.
        
        Args:
            layer_mapping: Mapping between student and teacher layer names
            weights: Weight for each layer pair's loss
            temperature: Temperature for attention/logit scaling
        """
        self.layer_mapping = layer_mapping
        self.weights = weights or {name: 1.0 for name in layer_mapping}
        self.temperature = temperature
        self.feature_distill = FeatureDistillation()
        self.attention_distill = AttentionDistillation(temperature)
        
    def __call__(
        self,
        student_features: Dict[str, Union[torch.Tensor, tf.Tensor]], # type: ignore
        teacher_features: Dict[str, Union[torch.Tensor, tf.Tensor]] # type: ignore
    ) -> Union[torch.Tensor, tf.Tensor]: # type: ignore
        """Calculate intermediate distillation loss."""
        if isinstance(next(iter(student_features.values())), torch.Tensor):
            return self._torch_loss(student_features, teacher_features)
        else:
            return self._tf_loss(student_features, teacher_features)
            
    def _torch_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """PyTorch intermediate distillation loss."""
        total_loss = 0.0
        
        for student_name, teacher_name in self.layer_mapping.items():
            s_feat = student_features[student_name]
            t_feat = teacher_features[teacher_name]
            weight = self.weights[student_name]
            
            # Feature distillation
            feat_loss = self.feature_distill(
                [s_feat],
                [t_feat]
            )
            
            # Attention distillation if feature maps have spatial dimensions
            if len(s_feat.shape) == 4:  # [B, C, H, W]
                s_att = self._compute_attention_map(s_feat)
                t_att = self._compute_attention_map(t_feat)
                att_loss = self.attention_distill(s_att, t_att)
                total_loss += weight * (feat_loss + att_loss)
            else:
                total_loss += weight * feat_loss
                
        return total_loss
        
    def _tf_loss(
        self,
        student_features: Dict[str, tf.Tensor], # type: ignore
        teacher_features: Dict[str, tf.Tensor] # type: ignore
    ) -> tf.Tensor: # type: ignore
        """TensorFlow intermediate distillation loss."""
        total_loss = 0.0
        
        for student_name, teacher_name in self.layer_mapping.items():
            s_feat = student_features[student_name]
            t_feat = teacher_features[teacher_name]
            weight = self.weights[student_name]
            
            # Feature distillation
            feat_loss = self.feature_distill(
                [s_feat],
                [t_feat]
            )
            
            # Attention distillation if feature maps have spatial dimensions
            if len(s_feat.shape) == 4:  # [B, C, H, W]
                s_att = self._compute_attention_map_tf(s_feat)
                t_att = self._compute_attention_map_tf(t_feat)
                att_loss = self.attention_distill(s_att, t_att)
                total_loss += weight * (feat_loss + att_loss)
            else:
                total_loss += weight * feat_loss
                
        return total_loss
        
    def _compute_attention_map(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial attention map from feature tensor.
        
        Generates attention maps by averaging feature activations across channels.
        This helps identify important spatial regions in the feature maps.
        
        Args:
            features: Input feature tensor
                Shape: [batch_size, channels, height, width]
                
        Returns:
            Spatial attention map
                Shape: [batch_size, height, width]
                
        Example:
            ```python
            feat = model.get_features(x)
            att_map = distiller._compute_attention_map(feat)
            
            # Visualize attention
            plt.imshow(att_map[0].cpu())
            ```
        """
        # [B, C, H, W] -> [B, H, W]
        return torch.mean(torch.abs(features), dim=1)
        
    def _compute_attention_map_tf(
        self,
        features: tf.Tensor # type: ignore
    ) -> tf.Tensor: # type: ignore
        """Compute spatial attention map from feature tensor.
        
        TensorFlow implementation of attention map computation. Generates
        attention maps by averaging feature activations across channels.
        
        Args:
            features: Input feature tensor
                Shape: [batch_size, height, width, channels]
                
        Returns:
            Spatial attention map
                Shape: [batch_size, height, width]
                
        Example:
            ```python
            feat = model.get_features(x)
            att_map = distiller._compute_attention_map_tf(feat)
            
            # Visualize attention
            plt.imshow(att_map[0])
            ```
        """
        # [B, H, W, C] -> [B, H, W]
        return tf.reduce_mean(tf.abs(features), axis=-1)

class RelationDistillation:
    """Relational knowledge distillation implementation.
    
    This class implements relational knowledge distillation which transfers
    structural knowledge by preserving instance relationships. Features:
    - Multiple distance metrics (euclidean, cosine)
    - Instance relationship preservation
    - Structure-aware distillation
    - Pairwise distance matching
    
    Example:
        ```python
        distiller = RelationDistillation(
            distance_type="cosine",
            temperature=4.0
        )
        
        loss = distiller(
            student_features=student_feat,
            teacher_features=teacher_feat
        )
        ```
    """

    def __init__(
        self,
        temperature: float = 1.0,
        distance_type: str = "euclidean"
    ):
        """Initialize relation distillation.
        
        Args:
            temperature: Temperature for softmax
            distance_type: Type of distance metric ('euclidean' or 'cosine')
        """
        self.temperature = temperature
        self.distance_type = distance_type
        
    def __call__(
        self,
        student_features: Union["torch.Tensor", "tf.Tensor"],  # type: ignore
        teacher_features: Union["torch.Tensor", "tf.Tensor"]   # type: ignore
    ) -> Union["torch.Tensor", "tf.Tensor"]:  # type: ignore
        """Calculate relation distillation loss."""
        if isinstance(student_features, torch.Tensor):
            return self._torch_loss(student_features, teacher_features)
        else:
            return self._tf_loss(student_features, teacher_features)
            
    def _torch_loss(
        self,
        student_features: "torch.Tensor",
        teacher_features: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute PyTorch relation distillation loss."""
        # Compute distance matrices
        s_dist = self._compute_distance_matrix_torch(student_features)
        t_dist = self._compute_distance_matrix_torch(teacher_features)
        
        # Convert to probability distributions
        s_prob = torch.softmax(s_dist / self.temperature, dim=-1)
        t_prob = torch.softmax(t_dist / self.temperature, dim=-1)
        
        # KL divergence loss
        loss = torch.nn.KLDivLoss(reduction='batchmean')(
            torch.log(s_prob),
            t_prob
        ) * (self.temperature ** 2)
        
        return loss
        
    def _tf_loss(
        self,
        student_features: "tf.Tensor",  # type: ignore
        teacher_features: "tf.Tensor"   # type: ignore
    ) -> "tf.Tensor":  # type: ignore
        """Compute TensorFlow relation distillation loss."""
        # Compute distance matrices
        s_dist = self._compute_distance_matrix_tf(student_features)
        t_dist = self._compute_distance_matrix_tf(teacher_features)
        
        # Convert to probability distributions
        s_prob = tf.nn.softmax(s_dist / self.temperature)
        t_prob = tf.nn.softmax(t_dist / self.temperature)
        
        # KL divergence loss
        loss = tf.keras.losses.KLDivergence()(
            t_prob,
            s_prob
        ) * (self.temperature ** 2)
        
        return loss
        
    def _compute_distance_matrix_torch(
        self,
        features: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute pairwise distance/similarity matrix for PyTorch tensors.
        
        Computes either Euclidean distances or cosine similarities between
        all pairs of feature vectors in the batch.
        
        Args:
            features: Input feature tensor
                Shape: [batch_size, num_instances, feature_dim]
                
        Returns:
            Distance/similarity matrix
                Shape: [batch_size, num_instances, num_instances]
                
        Example:
            ```python
            # Compute pairwise distances
            dist_mat = distiller._compute_distance_matrix_torch(features)
            
            # Convert to probabilities
            probs = torch.softmax(dist_mat / temperature, dim=-1)
            ```
        """
        if self.distance_type == "euclidean":
            # [B, N, D] -> [B, N, N]
            dist = torch.cdist(features, features)
        else:  # cosine
            # Normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            # Compute cosine similarity
            dist = torch.matmul(features, features.transpose(-2, -1))
            
        return dist
        
    def _compute_distance_matrix_tf(
        self,
        features: "tf.Tensor"  # type: ignore
    ) -> "tf.Tensor":  # type: ignore
        """Compute pairwise distance/similarity matrix for TensorFlow tensors.
        
        Computes either Euclidean distances or cosine similarities between
        all pairs of feature vectors in the batch.
        
        Args:
            features: Input feature tensor
                Shape: [batch_size, num_instances, feature_dim]
                
        Returns:
            Distance/similarity matrix
                Shape: [batch_size, num_instances, num_instances]
                
        Example:
            ```python
            # Compute pairwise distances
            dist_mat = distiller._compute_distance_matrix_tf(features)
            
            # Convert to probabilities
            probs = tf.nn.softmax(dist_mat / temperature)
            ```
        """
        if self.distance_type == "euclidean":
            # [B, N, D] -> [B, N, N]
            dist = tf.sqrt(tf.reduce_sum(
                (tf.expand_dims(features, 2) - tf.expand_dims(features, 1))**2,
                axis=-1
            ))
        else:  # cosine
            # Normalize features
            features = tf.nn.l2_normalize(features, axis=-1)
            # Compute cosine similarity
            dist = tf.matmul(features, features, transpose_b=True)
            
        return dist 

class AdversarialDistillation:
    """Adversarial knowledge distillation implementation.
    
    This class implements adversarial distillation where a discriminator network
    tries to distinguish between teacher and student features while the student
    tries to fool the discriminator. Features:
    - GAN-based feature distillation
    - Adversarial feature matching
    - Progressive distillation
    - Feature alignment loss
    
    Example:
        ```python
        distiller = AdversarialDistillation(
            teacher_model=teacher,
            student_model=student,
            discriminator=discriminator,
            alpha=0.5
        )
        
        # Train with adversarial distillation
        distiller.train(train_loader, epochs=10)
        ```
    """
    
    def __init__(
        self,
        teacher_model: Union["nn.Module", "tf.keras.Model"],  # type: ignore
        student_model: Union["nn.Module", "tf.keras.Model"],  # type: ignore
        discriminator: Union["nn.Module", "tf.keras.Model"],  # type: ignore
        alpha: float = 0.5
    ):
        """Initialize adversarial distillation.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            discriminator: Discriminator network
            alpha: Weight for adversarial loss component
        """
        self.teacher = teacher_model
        self.student = student_model
        self.discriminator = discriminator
        self.alpha = alpha
        self.is_torch = isinstance(teacher_model, nn.Module)

    def _tf_loss(
        self,
        student_features: "tf.Tensor",  # type: ignore
        teacher_features: "tf.Tensor",  # type: ignore
        is_training: bool
    ) -> "tf.Tensor":  # type: ignore
        """Compute TensorFlow adversarial distillation loss."""
        # Feature distillation loss
        feat_loss = self.feature_distill([student_features], [teacher_features])
        
        # Adversarial loss
        if is_training:
            # Train discriminator
            d_real = self.discriminator(teacher_features, training=True)
            d_fake = self.discriminator(student_features, training=True)
            d_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )(
                tf.ones_like(d_real),
                d_real
            ) + tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )(
                tf.zeros_like(d_fake),
                d_fake
            )
            
            # Train student (generator)
            d_fake = self.discriminator(student_features, training=True)
            g_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )(
                tf.ones_like(d_fake),
                d_fake
            )
            
            return feat_loss + self.alpha * (d_loss + g_loss)
        else:
            return feat_loss

    def _torch_loss(
        self,
        student_features: "torch.Tensor",
        teacher_features: "torch.Tensor",
        is_training: bool
    ) -> "torch.Tensor":
        """Compute PyTorch adversarial distillation loss."""
        # Feature distillation loss
        feat_loss = self.feature_distill([student_features], [teacher_features])
        
        # Adversarial loss
        if is_training:
            # Train discriminator
            d_real = self.discriminator(teacher_features, training=True)
            d_fake = self.discriminator(student_features, training=True)
            d_loss = torch.nn.BCELoss()(
                d_real,
                torch.ones_like(d_real)
            ) + torch.nn.BCELoss()(
                d_fake,
                torch.zeros_like(d_fake)
            )
            
            # Train student (generator)
            d_fake = self.discriminator(student_features, training=True)
            g_loss = torch.nn.BCELoss()(
                d_fake,
                torch.ones_like(d_fake)
            )
            
            return feat_loss + self.alpha * (d_loss + g_loss)
        else:
            return feat_loss

class KnowledgeDistillation:
    """Implements knowledge distillation training.
    
    This class handles the training process for knowledge distillation, including:
    - Computing soft targets from teacher model
    - Calculating distillation loss
    - Combining with standard cross-entropy loss
    - Monitoring distillation metrics
    
    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        config: Distillation configuration
    """

    def __init__(
        self,
        teacher_model: "nn.Module",
        student_model: "nn.Module",
        config: "DistillationConfig"
    ):
        """Initialize knowledge distillation trainer.
        
        Sets up the distillation process with teacher and student models
        and configures the training parameters.
        
        Args:
            teacher_model: Pre-trained teacher model
                Must be in eval mode and have its parameters frozen
            student_model: Student model to train
                Should have fewer parameters than teacher
            config: Distillation configuration including:
                - temperature: Softmax temperature
                - alpha: Loss balancing weight
                - method: Distillation method to use
                - feature_layers: Layers for feature matching
                - attention_layers: Layers for attention transfer
                
        Example:
            ```python
            distiller = KnowledgeDistillation(
                teacher_model=teacher,
                student_model=student,
                config=DistillationConfig(
                    temperature=4.0,
                    alpha=0.5,
                    method=DistillationMethod.SOFT_TARGETS
                )
            )
            ```
        """
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self._setup_loss_functions()
        self.metrics = DistillationMetrics()
        self.monitor = DistillationMonitor()

    def _setup_loss_functions(self):
        """Setup loss functions based on distillation method.
        
        Configures the appropriate loss computation method based on the selected
        distillation strategy. Supported methods include:
        - Soft targets (KL divergence)
        - Attention transfer
        - Feature matching
        - Relational distillation
        
        Raises:
            ValueError: If unsupported distillation method is specified
            
        Example:
            ```python
            distiller = KnowledgeDistillation(
                teacher_model=teacher,
                student_model=student,
                config=DistillationConfig(method=DistillationMethod.SOFT_TARGETS)
            )
            distiller._setup_loss_functions()
            ```
        """
        if self.config.method == DistillationMethod.SOFT_TARGETS:
            self.compute_loss = self._compute_soft_targets_loss
        elif self.config.method == DistillationMethod.ATTENTION:
            self.compute_loss = self._compute_attention_loss
        elif self.config.method == DistillationMethod.FEATURE_MATCHING:
            self.compute_loss = self._compute_feature_loss
        elif self.config.method == DistillationMethod.RELATION:
            self.compute_loss = self._compute_relation_loss
        else:
            raise ValueError(f"Unsupported distillation method: {self.config.method}")

    def _compute_soft_targets_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss with soft targets.
        
        Combines cross-entropy loss on hard targets with KL divergence loss
        between softened logits from teacher and student models.
        
        Args:
            student_logits: Raw logits from student model
                Shape: [batch_size, num_classes]
            teacher_logits: Raw logits from teacher model
                Shape: [batch_size, num_classes]
                
        Returns:
            Combined loss:
                alpha * KL(student_soft || teacher_soft) + 
                (1 - alpha) * CE(student_hard, targets)
                
        Example:
            ```python
            loss = distiller._compute_soft_targets_loss(
                student_out["logits"],
                teacher_out["logits"]
            )
            ```
        """
        T = self.config.temperature
        alpha = self.config.alpha
        
        # Soft targets loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, teacher_logits)
        
        return alpha * soft_loss + (1 - alpha) * hard_loss

    def _compute_attention_loss(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention transfer loss between models.
        
        Transfers attention patterns from teacher to student by minimizing
        the difference between their attention maps. This helps the student
        learn similar feature focusing behavior.
        
        Args:
            student_attention: Student model attention maps
                Shape: [batch_size, num_heads, seq_len, seq_len]
            teacher_attention: Teacher model attention maps
                Shape: [batch_size, num_heads, seq_len, seq_len]
                
        Returns:
            Attention transfer loss (MSE between normalized attention maps)
            
        Example:
            ```python
            s_att = student.get_attention_maps(x)
            t_att = teacher.get_attention_maps(x)
            att_loss = optimizer._compute_attention_loss(s_att, t_att)
            ```
        """
        return DistillationLoss.attention_transfer(
            student_attention,
            teacher_attention
        )

    def _compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature matching loss between teacher and student.
        
        Matches intermediate feature representations between corresponding
        layers of teacher and student models using L2 loss.
        
        Args:
            student_features: Dictionary of student model features
                Keys: Layer names
                Values: Feature tensors [batch_size, channels, height, width]
            teacher_features: Dictionary of teacher model features
                Keys: Layer names
                Values: Feature tensors [batch_size, channels, height, width]
                
        Returns:
            Average feature matching loss across all specified layers
            
        Example:
            ```python
            s_feats = student.get_features(x)
            t_feats = teacher.get_features(x)
            loss = distiller._compute_feature_loss(s_feats, t_feats)
            ```
        """
        loss = 0
        for layer in self.config.feature_layers:
            loss += DistillationLoss.feature_matching(
                student_features[layer],
                teacher_features[layer]
            )
        return loss / len(self.config.feature_layers)

    def _compute_relation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute relational knowledge distillation loss.
        
        Preserves structural relationships between instances by matching
        pairwise distance matrices between teacher and student outputs.
        
        Args:
            student_logits: Student model output logits
                Shape: [batch_size, num_classes]
            teacher_logits: Teacher model output logits
                Shape: [batch_size, num_classes]
                
        Returns:
            KL divergence loss between distance probability distributions
            
        Example:
            ```python
            loss = distiller._compute_relation_loss(
                student_out["logits"],
                teacher_out["logits"]
            )
            ```
        """
        # Compute distance matrices
        s_dist = self._compute_distance_matrix_torch(student_logits)
        t_dist = self._compute_distance_matrix_torch(teacher_logits)
        
        # Convert to probability distributions
        s_prob = torch.softmax(s_dist / self.config.temperature, dim=-1)
        t_prob = torch.softmax(t_dist / self.config.temperature, dim=-1)
        
        # KL divergence loss
        loss = torch.nn.KLDivLoss(reduction='batchmean')(
            torch.log(s_prob),
            t_prob
        ) * (self.config.temperature ** 2)
        
        return loss

    def _compute_distance_matrix_torch(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distance matrix for PyTorch tensors."""
        dist = torch.cdist(features, features)
        return dist

    def _get_layer_features(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate layer features from model.
        
        Performs forward pass and collects feature maps from specified
        intermediate layers of the model using hooks.
        
        Args:
            model: PyTorch model to extract features from
            inputs: Input tensor
                Shape: [batch_size, *input_shape]
            layer_names: Names of layers to extract features from
                Must match layer names in model.named_modules()
            
        Returns:
            Dictionary mapping layer names to feature tensors
                Keys: Layer names
                Values: Feature tensors [batch_size, channels, height, width]
            
        Example:
            ```python
            features = distiller._get_layer_features(
                model=student_model,
                inputs=batch_x,
                layer_names=["layer1", "layer2", "layer3"]
            )
            
            for name, feat in features.items():
                print(f"{name} shape: {feat.shape}")
            ```
        """
        features = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register forward hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
            
        # Forward pass
        model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features

    def _compute_metrics(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive distillation metrics.
        
        Calculates various metrics to monitor distillation progress:
        - Accuracy metrics
        - Loss components
        - Teacher agreement
        - Feature similarity
        
        Args:
            student_outputs: Dictionary of student model outputs
                Must contain 'logits' key with predictions
            teacher_outputs: Dictionary of teacher model outputs
                Must contain 'logits' key with predictions
            targets: Ground truth labels
                Shape: [batch_size]
            
        Returns:
            Dictionary containing metrics:
                - accuracy: Student model accuracy
                - teacher_accuracy: Teacher model accuracy
                - distillation_loss: Knowledge distillation loss
                - agreement: Prediction agreement with teacher
                - feature_similarity: Feature space similarity
            
        Example:
            ```python
            metrics = distiller._compute_metrics(
                student_outputs=student_out,
                teacher_outputs=teacher_out,
                targets=labels
            )
            print(f"Student accuracy: {metrics['accuracy']:.2%}")
            ```
        """
        metrics = {}
        
        # Accuracy metrics
        metrics["accuracy"] = self._compute_accuracy(student_outputs["logits"], targets)
        metrics["teacher_accuracy"] = self._compute_accuracy(teacher_outputs["logits"], targets)
        
        # Loss components
        metrics["distillation_loss"] = self.compute_loss(
            student_outputs["logits"],
            teacher_outputs["logits"]
        ).item()
        
        # Teacher agreement
        metrics["agreement"] = self._compute_agreement(
            student_outputs["logits"],
            teacher_outputs["logits"]
        )
        
        # Feature similarity if available
        if "features" in student_outputs and "features" in teacher_outputs:
            metrics["feature_similarity"] = self._compute_feature_similarity(
                student_outputs["features"],
                teacher_outputs["features"]
            )
        
        return metrics

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train one epoch with knowledge distillation.
        
        Performs training for one complete epoch including:
        - Forward passes through teacher and student
        - Loss computation and backpropagation
        - Optimizer steps
        - Metrics collection and averaging
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for student model
            
        Returns:
            Dictionary containing epoch metrics:
                - loss: Average training loss
                - accuracy: Average training accuracy
                - teacher_agreement: Average agreement with teacher
                - feature_similarity: Average feature similarity if available
            
        Example:
            ```python
            epoch_metrics = distiller._train_epoch(
                train_loader=train_loader,
                optimizer=optimizer
            )
            print(f"Epoch loss: {epoch_metrics['loss']:.4f}")
            ```
        """
        self.student.train()
        self.teacher.eval()
        
        total_metrics = defaultdict(float)
        num_batches = 0
        
        for inputs, targets in train_loader:
            # Forward pass
            student_outputs = self.student(inputs)
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
            
            # Compute loss and metrics
            loss = self.compute_loss(
                student_outputs["logits"],
                teacher_outputs["logits"]
            )
            metrics = self._compute_metrics(
                student_outputs,
                teacher_outputs,
                targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            metrics["loss"] = loss.item()
            for k, v in metrics.items():
                total_metrics[k] += v
            num_batches += 1
        
        # Calculate averages
        return {
            k: v / num_batches
            for k, v in total_metrics.items()
        }

    def _setup_optimizer(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        **kwargs
    ) -> torch.optim.Optimizer:
        """Setup optimizer for student model training.
        
        Configures the optimizer with specified parameters for training
        the student model during knowledge distillation.
        
        Args:
            lr: Learning rate (default: 0.001)
            weight_decay: Weight decay coefficient (default: 0.0)
            **kwargs: Additional optimizer arguments:
                - momentum: Momentum coefficient for SGD
                - beta1: Beta1 coefficient for Adam
                - beta2: Beta2 coefficient for Adam
                - optimizer_type: Type of optimizer ("adam", "sgd", etc.)
            
        Returns:
            Configured PyTorch optimizer
            
        Example:
            ```python
            optimizer = distiller._setup_optimizer(
                lr=0.001,
                weight_decay=1e-4,
                optimizer_type="adam",
                beta1=0.9,
                beta2=0.999
            )
            ```
        """
        optimizer_type = kwargs.pop("optimizer_type", "adam")
        
        if optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                self.student.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                self.student.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _setup_scheduler(
        self,
        optimizer: "torch.optim.Optimizer",
        **kwargs
    ) -> Optional["torch.optim.lr_scheduler._LRScheduler"]:
        """Setup learning rate scheduler for training.
        
        Configures learning rate scheduler to adjust learning rate during
        training based on specified schedule.
        
        Args:
            optimizer: Optimizer to schedule
            **kwargs: Scheduler configuration:
                - type: Scheduler type ("step", "cosine", "plateau", etc.)
                - step_size: Steps between learning rate updates
                - gamma: Learning rate decay factor
                - T_max: Number of iterations for cosine schedule
                - eta_min: Minimum learning rate for cosine schedule
            
        Returns:
            Configured learning rate scheduler or None if not specified
            
        Example:
            ```python
            scheduler = distiller._setup_scheduler(
                optimizer,
                type="cosine",
                T_max=10,
                eta_min=1e-6
            )
            ```
        """
        scheduler_type = kwargs.pop("type", None)
        
        if scheduler_type is None:
            return None
        
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 10),
                eta_min=kwargs.get("eta_min", 0)
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=kwargs.get("patience", 10),
                factor=kwargs.get("factor", 0.1)
            ) 
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _plot_progress(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Plot training progress and metrics.
        
        Visualizes training progress including:
        - Loss curves
        - Accuracy metrics
        - Teacher-student agreement
        - Learning rate changes
        
        Args:
            train_metrics: Training metrics for current epoch
            val_metrics: Optional validation metrics
            
        Example:
            ```python
            distiller._plot_progress(
                train_metrics={
                    "loss": 0.5,
                    "accuracy": 0.85,
                    "agreement": 0.90
                },
                val_metrics={
                    "loss": 0.6,
                    "accuracy": 0.82
                }
            )
            ```
        """
        # Plot loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.plot(train_metrics["loss"], label="Train")
        if val_metrics:
            plt.plot(val_metrics["loss"], label="Val")
        plt.title("Loss")
        plt.legend()
        
        # Plot accuracy
        plt.subplot(132)
        plt.plot(train_metrics["accuracy"], label="Train")
        if val_metrics:
            plt.plot(val_metrics["accuracy"], label="Val")
        plt.title("Accuracy")
        plt.legend()
        
        # Plot teacher agreement
        plt.subplot(133)
        plt.plot(train_metrics["agreement"], label="Train")
        if val_metrics and "agreement" in val_metrics:
            plt.plot(val_metrics["agreement"], label="Val")
        plt.title("Teacher Agreement")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def train_step(
        self,
        inputs: "torch.Tensor",
        targets: "torch.Tensor"
    ) -> Dict[str, float]:
        """Perform single training step with distillation.
        
        Executes one training iteration including:
        - Forward pass through teacher and student
        - Loss computation
        - Backward pass and optimization
        - Metrics collection
        
        Args:
            inputs: Input batch
                Shape: [batch_size, *input_shape]
            targets: Target labels
                Shape: [batch_size]
            
        Returns:
            Dictionary containing:
                - total_loss: Combined distillation loss
                - soft_loss: Soft targets loss component
                - hard_loss: Hard targets loss component
                - accuracy: Batch accuracy
            
        Example:
            ```python
            for batch in train_loader:
                inputs, targets = batch
                metrics = distiller.train_step(inputs, targets)
                print(f"Loss: {metrics['total_loss']:.4f}")
            ```
        """
        # Get model outputs
        student_outputs = self.student(inputs)
        teacher_outputs = self.teacher(inputs)
        
        # Compute loss and metrics
        loss, metrics = self.compute_loss(
            student_outputs["logits"],
            teacher_outputs["logits"]
        )
        
        # Update student model
        self.student.zero_grad()
        loss.backward()
        self.student.step()
        
        # Update metrics
        metrics["total_loss"] = loss.item()
        
        return metrics

    def distill(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        **kwargs
    ) -> nn.Module:
        """Train student model using knowledge distillation.
        
        Performs complete distillation training process including:
        - Temperature-scaled soft targets
        - Optional feature/attention matching
        - Progress monitoring and visualization
        - Early stopping if validation data provided
        - Adaptive temperature scheduling
        - Learning rate adjustment
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            **kwargs: Additional training arguments:
                - optimizer_config: Optimizer parameters
                    - lr: Learning rate
                    - weight_decay: Weight decay
                - scheduler_config: Learning rate scheduler
                    - type: Scheduler type (step, cosine, etc.)
                    - params: Scheduler parameters
                - early_stopping: Whether to use early stopping
                - patience: Early stopping patience
                - monitor: Metric to monitor
                - visualization: Whether to plot training progress
                - save_best: Whether to save best model
                - save_path: Path to save model checkpoints
            
        Returns:
            Trained student model
            
        Example:
            ```python
            student = distiller.distill(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,
                optimizer_config={
                    "lr": 0.001,
                    "weight_decay": 1e-4
                },
                scheduler_config={
                    "type": "cosine",
                    "params": {"T_max": 10}
                },
                early_stopping=True,
                patience=5,
                save_best=True,
                save_path="checkpoints/student.pt"
            )
            
            # Evaluate trained model
            test_metrics = distiller.evaluate(test_loader)
            print(f"Final accuracy: {test_metrics['accuracy']:.2%}")
            ```
            
        Notes:
            - The teacher model should be in eval mode
            - For best results, use a validation set and early stopping
            - Monitor training progress with visualization tools
            - Save checkpoints for model recovery
        """
        # Setup training components
        optimizer = self._setup_optimizer(**kwargs.get("optimizer_config", {}))
        scheduler = self._setup_scheduler(optimizer, **kwargs.get("scheduler_config", {}))
        
        # Initialize tracking
        best_metric = float("inf")
        patience_counter = 0
        best_model = None
        
        # Training loop
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                # Early stopping check
                if kwargs.get("early_stopping", False):
                    monitor_metric = val_metrics[kwargs.get("monitor", "loss")]
                    if monitor_metric < best_metric:
                        best_metric = monitor_metric
                        patience_counter = 0
                        if kwargs.get("save_best", False):
                            best_model = copy.deepcopy(self.student)
                    else:
                        patience_counter += 1
                        if patience_counter >= kwargs.get("patience", 5):
                            print(f"Early stopping at epoch {epoch}")
                            break
                            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Visualization
            if kwargs.get("visualization", False):
                self._plot_progress(train_metrics, val_metrics)
            
        # Return best model if saved
        if best_model is not None:
            return best_model
        return self.student

    def evaluate(
        self,
        test_loader: "torch.utils.data.DataLoader"
    ) -> Dict[str, float]:
        """Evaluate student model performance.
        
        Performs comprehensive evaluation including:
        - Model prediction
        - Metrics computation
        - Teacher-student comparison
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics:
                - accuracy: Test accuracy
                - distillation_loss: Distillation loss
                - teacher_agreement: Agreement with teacher
                - confidence: Prediction confidence
            
        Example:
            ```python
            metrics = distiller.evaluate(test_loader)
            print(f"Test Accuracy: {metrics['accuracy']:.2%}")
            ```
        """
        self.student.eval()
        self.teacher.eval()
        
        total_metrics = defaultdict(float)
        num_batches = 0
        
        for data, labels in test_loader:
            # Get model outputs
            student_outputs = self.student(data)
            teacher_outputs = self.teacher(data)
            
            # Compute metrics
            metrics = self.compute_loss(
                student_outputs["logits"],
                teacher_outputs["logits"]
            )
            
            # Update totals
            for k, v in metrics.items():
                total_metrics[k] += v
            num_batches += 1
        
        # Calculate averages
        return {
            k: v / num_batches
            for k, v in total_metrics.items()
        }

    def _compute_accuracy(
        self,
        logits: "torch.Tensor",
        targets: "torch.Tensor"
    ) -> float:
        """Compute prediction accuracy.
        
        Calculates classification accuracy by comparing predicted class
        with ground truth labels.
        
        Args:
            logits: Model output logits
                Shape: [batch_size, num_classes]
            targets: Ground truth labels
                Shape: [batch_size]
            
        Returns:
            Classification accuracy as a float between 0 and 1
            
        Example:
            ```python
            accuracy = distiller._compute_accuracy(
                student_outputs["logits"],
                batch_labels
            )
            print(f"Batch accuracy: {accuracy:.2%}")
            ```
        """
        predictions = torch.argmax(logits, dim=1)
        return (predictions == targets).float().mean().item()

    def _compute_agreement(
        self,
        student_logits: "torch.Tensor",
        teacher_logits: "torch.Tensor"
    ) -> float:
        """Compute prediction agreement between teacher and student.
        
        Calculates the percentage of predictions where teacher and student
        models predict the same class, regardless of correctness.
        
        Args:
            student_logits: Student model output logits
                Shape: [batch_size, num_classes]
            teacher_logits: Teacher model output logits
                Shape: [batch_size, num_classes]
            
        Returns:
            Agreement ratio as a float between 0 and 1
            
        Example:
            ```python
            agreement = distiller._compute_agreement(
                student_out["logits"],
                teacher_out["logits"]
            )
            print(f"Teacher agreement: {agreement:.2%}")
            ```
        """
        student_preds = torch.argmax(student_logits, dim=1)
        teacher_preds = torch.argmax(teacher_logits, dim=1)
        return (student_preds == teacher_preds).float().mean().item() 