from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .distillation import KnowledgeDistillation
from .metrics_collector import MetricsCollector
import json
import jinja2
from pathlib import Path

class AnalysisType(Enum):
    """Distillation analysis types."""
    LOSS_ANALYSIS = "loss"
    FEATURE_ANALYSIS = "feature"
    ATTENTION_ANALYSIS = "attention"
    GRADIENT_ANALYSIS = "gradient"
    REPRESENTATION_ANALYSIS = "representation"

@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    types: List[AnalysisType]
    n_samples: int = 100
    n_components: int = 3
    layer_names: Optional[List[str]] = None
    save_dir: str = "analysis"

class DistillationAnalyzer:
    """Analyzes knowledge distillation process."""
    
    def __init__(
        self,
        distillation: KnowledgeDistillation,
        config: AnalysisConfig
    ):
        """Initialize analyzer.
        
        Args:
            distillation: Knowledge distillation instance
            config: Analysis configuration
        """
        self.distillation = distillation
        self.config = config
        self.metrics = MetricsCollector()
        
    def analyze(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Run distillation analysis.
        
        Args:
            data_loader: Data loader for analysis
            
        Returns:
            Analysis results
        """
        results = {}
        
        for analysis_type in self.config.types:
            if analysis_type == AnalysisType.LOSS_ANALYSIS:
                results["loss"] = self._analyze_loss(data_loader)
            elif analysis_type == AnalysisType.FEATURE_ANALYSIS:
                results["feature"] = self._analyze_features(data_loader)
            elif analysis_type == AnalysisType.ATTENTION_ANALYSIS:
                results["attention"] = self._analyze_attention(data_loader)
            elif analysis_type == AnalysisType.GRADIENT_ANALYSIS:
                results["gradient"] = self._analyze_gradients(data_loader)
            else:  # REPRESENTATION_ANALYSIS
                results["representation"] = self._analyze_representations(data_loader)
                
        return results
        
    def _analyze_loss(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Analyze loss behavior."""
        losses = {
            "ce_loss": [],
            "distill_loss": [],
            "total_loss": []
        }
        
        for data, labels in data_loader:
            # Get model outputs
            student_outputs = self.distillation._forward_student(data)
            teacher_outputs = self.distillation._forward_teachers(data)
            
            # Compute loss
            loss, metrics = self.distillation.compute_loss(
                student_outputs,
                teacher_outputs,
                labels
            )
            
            for k, v in metrics.items():
                losses[k].append(v)
                
            if len(losses["ce_loss"]) >= self.config.n_samples:
                break
                
        return {
            "statistics": {
                k: {
                    "mean": np.mean(v),
                    "std": np.std(v),
                    "min": np.min(v),
                    "max": np.max(v)
                }
                for k, v in losses.items()
            },
            "correlations": pd.DataFrame(losses).corr().to_dict(),
            "history": losses
        }
        
    def _analyze_features(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Analyze feature representations."""
        features = {
            "student": [],
            "teacher": []
        }
        
        for data, _ in data_loader:
            # Get model outputs
            student_outputs = self.distillation._forward_student(data)
            teacher_outputs = self.distillation._forward_teachers(data)
            
            # Extract features
            for layer in self.config.layer_names or []:
                student_feat = student_outputs["features"][layer]
                teacher_feat = teacher_outputs[0]["features"][layer]
                
                features["student"].append(student_feat.cpu().numpy())
                features["teacher"].append(teacher_feat.cpu().numpy())
                
            if len(features["student"]) >= self.config.n_samples:
                break
                
        return {
            "similarity": self._compute_feature_similarity(features),
            "statistics": self._compute_feature_statistics(features),
            "visualization": self._visualize_features(features)
        }
        
    def _analyze_attention(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Analyze attention patterns."""
        attention = {
            "student": [],
            "teacher": []
        }
        
        for data, _ in data_loader:
            # Get model outputs
            student_outputs = self.distillation._forward_student(data)
            teacher_outputs = self.distillation._forward_teachers(data)
            
            # Extract attention maps
            student_att = student_outputs.get("attention")
            teacher_att = teacher_outputs[0].get("attention")
            
            if student_att is not None and teacher_att is not None:
                attention["student"].append(student_att.cpu().numpy())
                attention["teacher"].append(teacher_att.cpu().numpy())
                
            if len(attention["student"]) >= self.config.n_samples:
                break
                
        return {
            "similarity": self._compute_attention_similarity(attention),
            "statistics": self._compute_attention_statistics(attention),
            "visualization": self._visualize_attention(attention)
        }
        
    def _analyze_gradients(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Analyze gradient flow."""
        gradients = []
        param_names = []
        
        for name, param in self.distillation.student.named_parameters():
            if param.requires_grad:
                param_names.append(name)
                
        for data, labels in data_loader:
            # Forward and backward pass
            student_outputs = self.distillation._forward_student(data)
            teacher_outputs = self.distillation._forward_teachers(data)
            
            loss, _ = self.distillation.compute_loss(
                student_outputs,
                teacher_outputs,
                labels
            )
            
            loss.backward()
            
            # Collect gradients
            grads = []
            for name in param_names:
                param = dict(self.distillation.student.named_parameters())[name]
                if param.grad is not None:
                    grads.append(param.grad.cpu().numpy())
                    
            gradients.append(grads)
            
            if len(gradients) >= self.config.n_samples:
                break
                
        return {
            "statistics": self._compute_gradient_statistics(gradients, param_names),
            "visualization": self._visualize_gradients(gradients, param_names)
        }
        
    def _analyze_representations(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Analyze learned representations."""
        representations = {
            "student": [],
            "teacher": []
        }
        labels = []
        
        for data, target in data_loader:
            # Get model outputs
            student_outputs = self.distillation._forward_student(data)
            teacher_outputs = self.distillation._forward_teachers(data)
            
            # Extract representations
            student_rep = student_outputs["logits"].cpu().numpy()
            teacher_rep = teacher_outputs[0]["logits"].cpu().numpy()
            
            representations["student"].append(student_rep)
            representations["teacher"].append(teacher_rep)
            labels.extend(target.cpu().numpy())
            
            if len(representations["student"]) >= self.config.n_samples:
                break
                
        return {
            "similarity": self._compute_representation_similarity(representations),
            "clustering": self._analyze_representation_clustering(
                representations,
                labels
            ),
            "visualization": self._visualize_representations(
                representations,
                labels
            )
        }
        
    def plot_analysis(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Generate analysis plots."""
        plots = {}
        
        for analysis_type, result in results.items():
            if analysis_type == "loss":
                plots["loss_trends"] = self._plot_loss_trends(result)
                plots["loss_correlations"] = self._plot_loss_correlations(result)
            elif analysis_type == "feature":
                plots["feature_similarity"] = self._plot_feature_similarity(result)
                plots["feature_distribution"] = self._plot_feature_distribution(result)
            elif analysis_type == "attention":
                plots["attention_patterns"] = self._plot_attention_patterns(result)
                plots["attention_similarity"] = self._plot_attention_similarity(result)
            elif analysis_type == "gradient":
                plots["gradient_flow"] = self._plot_gradient_flow(result)
                plots["gradient_distribution"] = self._plot_gradient_distribution(result)
            else:  # representation
                plots["representation_space"] = self._plot_representation_space(result)
                plots["representation_clustering"] = self._plot_representation_clustering(result)
                
        return plots 

    def _compute_feature_similarity(
        self,
        features: Dict[str, List[np.ndarray]]
    ) -> Dict[str, float]:
        """Compute feature similarity metrics."""
        similarities = {}
        
        # Flatten features
        student_feats = np.concatenate([f.reshape(f.shape[0], -1) for f in features["student"]])
        teacher_feats = np.concatenate([f.reshape(f.shape[0], -1) for f in features["teacher"]])
        
        # Compute cosine similarity
        student_norm = np.linalg.norm(student_feats, axis=1, keepdims=True)
        teacher_norm = np.linalg.norm(teacher_feats, axis=1, keepdims=True)
        cosine_sim = np.mean(
            np.sum(student_feats * teacher_feats, axis=1) / (student_norm * teacher_norm)
        )
        similarities["cosine"] = cosine_sim
        
        # Compute CKA similarity
        cka_sim = self._compute_cka_similarity(student_feats, teacher_feats)
        similarities["cka"] = cka_sim
        
        return similarities
        
    def _compute_feature_statistics(
        self,
        features: Dict[str, List[np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute feature statistics."""
        stats = {"student": {}, "teacher": {}}
        
        for model in ["student", "teacher"]:
            feats = features[model]
            if not feats:
                continue
            
            # Compute statistics
            feat_array = np.concatenate([f.reshape(f.shape[0], -1) for f in feats])
            stats[model] = {
                "mean": float(np.mean(feat_array)),
                "std": float(np.std(feat_array)),
                "sparsity": float(np.mean(feat_array == 0)),
                "min": float(np.min(feat_array)),
                "max": float(np.max(feat_array))
            }
        
        return stats
        
    def _compute_attention_similarity(
        self,
        attention: Dict[str, List[np.ndarray]]
    ) -> Dict[str, float]:
        """Compute attention similarity metrics."""
        similarities = {}
        
        if not attention["student"] or not attention["teacher"]:
            return similarities
        
        # Flatten attention maps
        student_att = np.concatenate([a.reshape(a.shape[0], -1) for a in attention["student"]])
        teacher_att = np.concatenate([a.reshape(a.shape[0], -1) for a in attention["teacher"]])
        
        # Compute KL divergence
        student_att = student_att / np.sum(student_att, axis=1, keepdims=True)
        teacher_att = teacher_att / np.sum(teacher_att, axis=1, keepdims=True)
        kl_div = np.mean(np.sum(
            teacher_att * np.log(teacher_att / (student_att + 1e-10)),
            axis=1
        ))
        similarities["kl_divergence"] = float(kl_div)
        
        # Compute attention correlation
        correlation = np.mean([
            np.corrcoef(s.flatten(), t.flatten())[0, 1]
            for s, t in zip(attention["student"], attention["teacher"])
        ])
        similarities["correlation"] = float(correlation)
        
        return similarities
        
    def _compute_attention_statistics(
        self,
        attention: Dict[str, List[np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute attention statistics."""
        stats = {"student": {}, "teacher": {}}
        
        for model in ["student", "teacher"]:
            att = attention[model]
            if not att:
                continue
            
            # Compute statistics
            att_array = np.concatenate([a.reshape(a.shape[0], -1) for a in att])
            stats[model] = {
                "entropy": float(np.mean(-np.sum(
                    att_array * np.log(att_array + 1e-10),
                    axis=1
                ))),
                "sparsity": float(np.mean(att_array < 0.01)),
                "max_attention": float(np.mean(np.max(att_array, axis=1))),
                "attention_spread": float(np.mean(np.std(att_array, axis=1)))
            }
        
        return stats
        
    def _compute_gradient_statistics(
        self,
        gradients: List[List[np.ndarray]],
        param_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute gradient statistics."""
        stats = {}
        
        for i, name in enumerate(param_names):
            layer_grads = np.stack([g[i].flatten() for g in gradients])
            stats[name] = {
                "mean": float(np.mean(layer_grads)),
                "std": float(np.std(layer_grads)),
                "norm": float(np.mean(np.linalg.norm(layer_grads, axis=1))),
                "zero_fraction": float(np.mean(np.abs(layer_grads) < 1e-10))
            }
        
        return stats
        
    def _compute_representation_similarity(
        self,
        representations: Dict[str, List[np.ndarray]]
    ) -> Dict[str, float]:
        """Compute representation similarity metrics."""
        similarities = {}
        
        # Concatenate representations
        student_reps = np.concatenate(representations["student"])
        teacher_reps = np.concatenate(representations["teacher"])
        
        # Compute MMD distance
        mmd = self._compute_mmd(student_reps, teacher_reps)
        similarities["mmd"] = float(mmd)
        
        # Compute Wasserstein distance
        wasserstein = self._compute_wasserstein(student_reps, teacher_reps)
        similarities["wasserstein"] = float(wasserstein)
        
        return similarities
        
    def _compute_cka_similarity(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """Compute CKA (Centered Kernel Alignment) similarity."""
        def center(K):
            n = K.shape[0]
            unit = np.ones([n, n])
            I = np.eye(n)
            H = I - unit / n
            return np.dot(np.dot(H, K), H)
        
        def rbf(X, Y, sigma=None):
            GX = np.dot(X, X.T)
            GY = np.dot(Y, Y.T)
            Q = np.tile(np.diag(GX), (GX.shape[0], 1))
            R = np.tile(np.diag(GY), (GY.shape[0], 1))
            H = Q + R.T - 2 * GX
            if sigma is None:
                sigma = np.median(H)
            return np.exp(-H / (2 * sigma**2))
        
        K = rbf(X, X)
        L = rbf(Y, Y)
        
        K = center(K)
        L = center(L)
        
        return np.sum(K * L) / np.sqrt(np.sum(K * K) * np.sum(L * L))
        
    def _compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sigma: float = 1.0
    ) -> float:
        """Compute Maximum Mean Discrepancy."""
        def gaussian_kernel(X, Y, sigma):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            
            return (
                np.exp(-XX / (2 * sigma**2)).mean() +
                np.exp(-YY / (2 * sigma**2)).mean() -
                2 * np.exp(-XY / (2 * sigma**2)).mean()
            )
        
        return gaussian_kernel(X, Y, sigma)
        
    def _compute_wasserstein(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        p: int = 2
    ) -> float:
        """Compute approximate Wasserstein distance."""
        # Use means and covariances as proxy
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        X_cov = np.cov(X.T)
        Y_cov = np.cov(Y.T)
        
        mean_term = np.linalg.norm(X_mean - Y_mean, ord=p)
        cov_term = np.linalg.norm(X_cov - Y_cov, ord='fro')
        
        return mean_term + cov_term 

    def _plot_loss_trends(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot loss trends."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss Values', 'Loss Ratios')
        )
        
        # Plot loss values
        for loss_type, values in result["history"].items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    name=loss_type,
                    mode="lines"
                ),
                row=1, col=1
            )
            
        # Plot loss ratios
        ce_loss = np.array(result["history"]["ce_loss"])
        distill_loss = np.array(result["history"]["distill_loss"])
        ratio = distill_loss / (ce_loss + 1e-10)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ratio))),
                y=ratio,
                name="distill/ce ratio",
                mode="lines"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title="Loss Analysis",
            showlegend=True
        )
        
        return fig
        
    def _plot_loss_correlations(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot loss correlations."""
        corr = pd.DataFrame(result["correlations"])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Loss Correlations",
            xaxis_title="Loss Type",
            yaxis_title="Loss Type"
        )
        
        return fig
        
    def _plot_feature_similarity(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot feature similarity analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Layer-wise Similarity', 'Feature Distribution')
        )
        
        # Plot layer-wise similarity
        similarities = result["cross_layer_similarity"]["matrix"]
        student_layers = result["cross_layer_similarity"]["student_layers"]
        teacher_layers = result["cross_layer_similarity"]["teacher_layers"]
        
        fig.add_trace(
            go.Heatmap(
                z=similarities,
                x=teacher_layers,
                y=student_layers,
                colorscale="Viridis",
                colorbar=dict(title="CKA Similarity")
            ),
            row=1, col=1
        )
        
        # Plot feature distribution
        for model in ["student", "teacher"]:
            stats = result["statistics"][model]
            fig.add_trace(
                go.Box(
                    y=list(stats.values()),
                    name=model,
                    boxpoints="all"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            title="Feature Analysis",
            showlegend=True
        )
        
        return fig
        
    def _plot_feature_distribution(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot feature distributions."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Student Features", "Teacher Features")
        )
        
        stats = result["statistics"]
        for i, model in enumerate(["student", "teacher"], 1):
            model_stats = stats[model]
            fig.add_trace(
                go.Box(
                    y=list(model_stats.values()),
                    name=model,
                    boxpoints="all"
                ),
                row=1,
                col=i
            )
            
        fig.update_layout(
            title="Feature Statistics Distribution",
            showlegend=True
        )
        
        return fig
        
    def _plot_attention_patterns(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot attention pattern analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Attention Distribution',
                'Attention Sparsity',
                'Head Importance',
                'Cross-Attention Similarity'
            )
        )
        
        # Plot attention distribution
        for model in ["student", "teacher"]:
            stats = result["statistics"][model]
            fig.add_trace(
                go.Violin(
                    y=stats["attention_values"],
                    name=model,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=1
            )
        
        # Plot attention sparsity
        sparsity = {
            model: stats["sparsity"]
            for model, stats in result["statistics"].items()
        }
        fig.add_trace(
            go.Bar(
                x=list(sparsity.keys()),
                y=list(sparsity.values()),
                name="sparsity"
            ),
            row=1, col=2
        )
        
        # Plot head importance
        importance = result.get("head_importance", {})
        for model in ["student", "teacher"]:
            if model in importance:
                fig.add_trace(
                    go.Bar(
                        x=[f"head_{i}" for i in range(len(importance[model]))],
                        y=importance[model],
                        name=f"{model}_heads"
                    ),
                    row=2, col=1
                )
        
        # Plot cross-attention similarity
        similarity = result["similarity"]
        fig.add_trace(
            go.Bar(
                x=list(similarity.keys()),
                y=list(similarity.values()),
                name="similarity"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Attention Analysis",
            showlegend=True
        )
        
        return fig
        
    def _plot_attention_similarity(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot attention similarity metrics."""
        fig = go.Figure()
        
        metrics = result["similarity"]
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values())
        ))
        
        fig.update_layout(
            title="Attention Similarity Metrics",
            xaxis_title="Metric",
            yaxis_title="Value"
        )
        
        return fig
        
    def _plot_gradient_flow(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot gradient flow analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Gradient Norms',
                'Zero Gradients',
                'Gradient Distribution',
                'Layer Updates'
            )
        )
        
        stats = result["statistics"]
        layers = list(stats.keys())
        
        # Plot gradient norms
        norms = [s["norm"] for s in stats.values()]
        fig.add_trace(
            go.Bar(
                x=layers,
                y=norms,
                name="gradient_norm"
            ),
            row=1, col=1
        )
        
        # Plot zero gradient fraction
        zero_fracs = [s["zero_fraction"] for s in stats.values()]
        fig.add_trace(
            go.Bar(
                x=layers,
                y=zero_fracs,
                name="zero_fraction"
            ),
            row=1, col=2
        )
        
        # Plot gradient distribution
        for i, layer in enumerate(layers):
            fig.add_trace(
                go.Box(
                    y=result["distributions"][layer],
                    name=layer,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot layer updates
        updates = result.get("updates", {})
        if updates:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(updates["magnitudes"]))),
                    y=updates["magnitudes"],
                    name="update_magnitude",
                    mode="lines"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Gradient Analysis",
            showlegend=True
        )
        
        return fig
        
    def _plot_gradient_distribution(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot gradient distribution."""
        fig = make_subplots(rows=2, cols=1)
        
        stats = result["statistics"]
        layers = list(stats.keys())
        
        # Mean and std
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=[s["mean"] for s in stats.values()],
                name="Mean",
                error_y=dict(
                    type="data",
                    array=[s["std"] for s in stats.values()]
                )
            ),
            row=1,
            col=1
        )
        
        # Zero fraction
        fig.add_trace(
            go.Bar(
                x=layers,
                y=[s["zero_fraction"] for s in stats.values()],
                name="Zero Fraction"
            ),
            row=2,
            col=1
        )
        
        fig.update_layout(
            title="Gradient Distribution Analysis",
            showlegend=True
        )
        
        return fig
        
    def _plot_representation_space(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot representation space analysis."""
        from sklearn.manifold import TSNE
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Representation Space (t-SNE)',
                'Representation Stability',
                'Feature Importance',
                'Adversarial Robustness'
            )
        )
        
        # Plot t-SNE visualization
        for model in ["student", "teacher"]:
            reps = np.array(result["reduced_representations"][model])
            if reps.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42)
                reps = tsne.fit_transform(reps)
                
            fig.add_trace(
                go.Scatter(
                    x=reps[:, 0],
                    y=reps[:, 1],
                    mode="markers",
                    name=model,
                    marker=dict(size=5)
                ),
                row=1, col=1
            )
        
        # Plot representation stability
        stability = result.get("stability", {})
        for model in ["student", "teacher"]:
            if model in stability:
                changes = stability[model]["changes"]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(changes))),
                        y=changes,
                        name=f"{model}_stability",
                        mode="lines"
                    ),
                    row=1, col=2
                )
        
        # Plot feature importance
        importance = result.get("feature_importance", {})
        for model in ["student", "teacher"]:
            if model in importance:
                top_features = importance[model]["top_features"]["rf"]
                scores = importance[model]["random_forest"]
                fig.add_trace(
                    go.Bar(
                        x=[f"feat_{i}" for i in top_features],
                        y=[scores[i] for i in top_features],
                        name=f"{model}_importance"
                    ),
                    row=2, col=1
                )
        
        # Plot adversarial robustness
        robustness = result.get("robustness", {})
        for model in ["student", "teacher"]:
            if model in robustness:
                changes = []
                layers = []
                for layer, stats in robustness[model].items():
                    changes.append(stats["mean_change"])
                    layers.append(layer)
                    
                fig.add_trace(
                    go.Bar(
                        x=layers,
                        y=changes,
                        name=f"{model}_robustness"
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title="Representation Analysis",
            showlegend=True
        )
        
        return fig
        
    def _plot_representation_clustering(
        self,
        result: Dict[str, Any]
    ) -> go.Figure:
        """Plot representation clustering."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Student Clusters", "Teacher Clusters")
        )
        
        # Add clustering plots
        # This is a placeholder - actual implementation would depend on
        # clustering method used
        
        fig.update_layout(
            title="Representation Clustering Analysis",
            showlegend=True
        )
        
        return fig

    def _analyze_representation_clustering(
        self,
        representations: Dict[str, List[np.ndarray]],
        labels: List[int]
    ) -> Dict[str, Any]:
        """Analyze representation clustering."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.decomposition import PCA
        
        results = {"student": {}, "teacher": {}}
        
        for model in ["student", "teacher"]:
            # Concatenate representations
            reps = np.concatenate(representations[model])
            
            # Reduce dimensionality for clustering
            if reps.shape[1] > 50:
                pca = PCA(n_components=50)
                reps = pca.fit_transform(reps)
                
            # Perform clustering
            n_clusters = len(np.unique(labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(reps)
            
            # Compute clustering metrics
            results[model] = {
                "silhouette": float(silhouette_score(reps, cluster_labels)),
                "calinski": float(calinski_harabasz_score(reps, cluster_labels)),
                "inertia": float(kmeans.inertia_),
                "cluster_sizes": [
                    int((cluster_labels == i).sum())
                    for i in range(n_clusters)
                ],
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "reduced_representations": reps.tolist()
            }
            
            # Compute alignment with true labels
            from sklearn.metrics import normalized_mutual_info_score
            results[model]["label_alignment"] = float(
                normalized_mutual_info_score(labels, cluster_labels)
            )
            
        return results

    def _analyze_layer_relationships(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze relationships between layers."""
        relationships = {}
        
        # Get layer names
        student_layers = list(student_outputs["features"].keys())
        teacher_layers = list(teacher_outputs["features"].keys())
        
        # Compute cross-layer similarities
        similarities = np.zeros((len(student_layers), len(teacher_layers)))
        
        for i, s_layer in enumerate(student_layers):
            for j, t_layer in enumerate(teacher_layers):
                s_feat = student_outputs["features"][s_layer]
                t_feat = teacher_outputs["features"][t_layer]
                
                # Flatten features
                s_feat = s_feat.reshape(s_feat.shape[0], -1)
                t_feat = t_feat.reshape(t_feat.shape[0], -1)
                
                # Compute CKA similarity
                similarities[i, j] = self._compute_cka_similarity(
                    s_feat.cpu().numpy(),
                    t_feat.cpu().numpy()
                )
                
        relationships["cross_layer_similarity"] = {
            "matrix": similarities.tolist(),
            "student_layers": student_layers,
            "teacher_layers": teacher_layers
        }
        
        # Find best matching layers
        best_matches = {}
        for i, s_layer in enumerate(student_layers):
            j = similarities[i].argmax()
            best_matches[s_layer] = {
                "teacher_layer": teacher_layers[j],
                "similarity": float(similarities[i, j])
            }
            
        relationships["best_matches"] = best_matches
        
        return relationships

    def _analyze_feature_importance(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze feature importance."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        importance = {}
        
        for model, outputs in [
            ("student", student_outputs),
            ("teacher", teacher_outputs)
        ]:
            model_importance = {}
            
            for layer, features in outputs["features"].items():
                # Flatten features
                feat = features.reshape(features.shape[0], -1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Train random forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(feat, labels_np)
                
                # Get feature importance scores
                rf_importance = rf.feature_importances_
                
                # Compute mutual information
                mi_importance = mutual_info_classif(feat, labels_np)
                
                model_importance[layer] = {
                    "random_forest": rf_importance.tolist(),
                    "mutual_info": mi_importance.tolist(),
                    "top_features": {
                        "rf": np.argsort(rf_importance)[-10:].tolist(),
                        "mi": np.argsort(mi_importance)[-10:].tolist()
                    }
                }
                
            importance[model] = model_importance
            
        return importance

    def _analyze_representation_stability(
        self,
        representations: Dict[str, List[np.ndarray]],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze representation stability over time."""
        stability = {}
        
        for model in ["student", "teacher"]:
            reps = representations[model]
            if len(reps) < window_size:
                continue
            
            # Compute representation changes
            changes = []
            for i in range(len(reps) - window_size):
                window = reps[i:i + window_size]
                
                # Compute average pairwise distance in window
                distances = []
                for j in range(window_size):
                    for k in range(j + 1, window_size):
                        dist = np.mean(np.abs(window[j] - window[k]))
                        distances.append(dist)
                        
                changes.append(np.mean(distances))
                
            stability[model] = {
                "mean_change": float(np.mean(changes)),
                "std_change": float(np.std(changes)),
                "change_trend": float(np.polyfit(
                    range(len(changes)),
                    changes,
                    deg=1
                )[0]),
                "changes": changes
            }
            
        return stability

    def _analyze_adversarial_robustness(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        epsilon: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze adversarial robustness of representations."""
        robustness = {}
        
        for model, outputs in [
            ("student", student_outputs),
            ("teacher", teacher_outputs)
        ]:
            model_robustness = {}
            
            for layer, features in outputs["features"].items():
                # Generate random perturbations
                perturb = torch.randn_like(features) * epsilon
                
                # Compute perturbed features
                perturbed_features = features + perturb
                
                # Compute relative change
                rel_change = torch.norm(
                    perturbed_features - features,
                    dim=1
                ) / torch.norm(features, dim=1)
                
                model_robustness[layer] = {
                    "mean_change": float(rel_change.mean()),
                    "std_change": float(rel_change.std()),
                    "max_change": float(rel_change.max()),
                    "quantiles": [
                        float(q) for q in torch.quantile(
                            rel_change,
                            torch.tensor([0.25, 0.5, 0.75])
                        )
                    ]
                }
                
            robustness[model] = model_robustness
            
        return robustness

    def _analyze_knowledge_transfer(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze knowledge transfer effectiveness."""
        transfer = {}
        
        # Get predictions
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs[0]["logits"]
        
        # Compute prediction agreement
        student_preds = torch.argmax(student_logits, dim=1)
        teacher_preds = torch.argmax(teacher_logits, dim=1)
        agreement = (student_preds == teacher_preds).float().mean().item()
        
        # Compute confidence calibration
        student_probs = torch.softmax(student_logits, dim=1)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        
        student_conf = student_probs.max(dim=1)[0]
        teacher_conf = teacher_probs.max(dim=1)[0]
        
        # Analyze confidence distribution
        transfer["confidence"] = {
            "student": {
                "mean": float(student_conf.mean()),
                "std": float(student_conf.std()),
                "histogram": self._compute_histogram(student_conf.cpu().numpy())
            },
            "teacher": {
                "mean": float(teacher_conf.mean()),
                "std": float(teacher_conf.std()),
                "histogram": self._compute_histogram(teacher_conf.cpu().numpy())
            }
        }
        
        # Analyze prediction agreement
        transfer["agreement"] = {
            "overall": agreement,
            "by_confidence": self._analyze_agreement_by_confidence(
                student_preds, teacher_preds,
                student_conf, teacher_conf
            ),
            "by_class": self._analyze_agreement_by_class(
                student_preds, teacher_preds,
                labels
            )
        }
        
        # Analyze knowledge distillation loss
        transfer["distillation_loss"] = {
            "kl_div": float(F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction='batchmean'
            )),
            "cross_entropy": float(F.cross_entropy(
                student_logits,
                labels
            ))
        }
        
        return transfer

    def _analyze_agreement_by_confidence(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        student_conf: torch.Tensor,
        teacher_conf: torch.Tensor,
        bins: int = 10
    ) -> Dict[str, List[float]]:
        """Analyze prediction agreement by confidence level."""
        agreement_by_conf = {
            "student": [],
            "teacher": []
        }
        
        # Analyze by student confidence
        student_conf_np = student_conf.cpu().numpy()
        for i in range(bins):
            lower = i / bins
            upper = (i + 1) / bins
            mask = (student_conf_np >= lower) & (student_conf_np < upper)
            if mask.any():
                agreement = (
                    student_preds[mask] == teacher_preds[mask]
                ).float().mean().item()
                agreement_by_conf["student"].append(agreement)
            
        # Analyze by teacher confidence
        teacher_conf_np = teacher_conf.cpu().numpy()
        for i in range(bins):
            lower = i / bins
            upper = (i + 1) / bins
            mask = (teacher_conf_np >= lower) & (teacher_conf_np < upper)
            if mask.any():
                agreement = (
                    student_preds[mask] == teacher_preds[mask]
                ).float().mean().item()
                agreement_by_conf["teacher"].append(agreement)
            
        return agreement_by_conf

    def _analyze_agreement_by_class(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[int, float]:
        """Analyze prediction agreement by class."""
        agreement_by_class = {}
        
        for class_idx in torch.unique(labels):
            mask = labels == class_idx
            if mask.any():
                agreement = (
                    student_preds[mask] == teacher_preds[mask]
                ).float().mean().item()
                agreement_by_class[int(class_idx)] = agreement
            
        return agreement_by_class

    def _analyze_ensemble_behavior(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Analyze ensemble behavior of teachers."""
        ensemble = {}
        
        # Get predictions
        student_logits = student_outputs["logits"]
        teacher_logits = [t["logits"] for t in teacher_outputs]
        
        # Compute teacher agreement
        teacher_preds = [torch.argmax(t, dim=1) for t in teacher_logits]
        teacher_probs = [torch.softmax(t, dim=1) for t in teacher_logits]
        
        # Analyze prediction diversity
        ensemble["diversity"] = self._analyze_prediction_diversity(
            teacher_preds,
            teacher_probs
        )
        
        # Analyze student-ensemble relationship
        ensemble["student_relationship"] = self._analyze_student_ensemble_relationship(
            student_logits,
            teacher_logits
        )
        
        return ensemble

    def _analyze_prediction_diversity(
        self,
        teacher_preds: List[torch.Tensor],
        teacher_probs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze prediction diversity among teachers."""
        n_teachers = len(teacher_preds)
        diversity = {}
        
        # Compute pairwise disagreement
        disagreement_matrix = torch.zeros(n_teachers, n_teachers)
        for i in range(n_teachers):
            for j in range(i + 1, n_teachers):
                disagreement = (
                    teacher_preds[i] != teacher_preds[j]
                ).float().mean().item()
                disagreement_matrix[i, j] = disagreement
                disagreement_matrix[j, i] = disagreement
            
        diversity["pairwise_disagreement"] = disagreement_matrix.tolist()
        
        # Compute ensemble entropy
        mean_probs = torch.mean(torch.stack(teacher_probs), dim=0)
        ensemble_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10),
            dim=1
        ).mean().item()
        
        diversity["ensemble_entropy"] = ensemble_entropy
        
        return diversity

    def _analyze_student_ensemble_relationship(
        self,
        student_logits: torch.Tensor,
        teacher_logits: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze relationship between student and teacher ensemble."""
        relationship = {}
        
        # Compute mean teacher predictions
        mean_teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
        
        # Compute KL divergence to ensemble
        relationship["kl_to_ensemble"] = float(F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(mean_teacher_logits, dim=1),
            reduction='batchmean'
        ))
        
        # Compute correlation with ensemble
        student_probs = F.softmax(student_logits, dim=1)
        mean_teacher_probs = F.softmax(mean_teacher_logits, dim=1)
        
        correlation = torch.corrcoef(torch.stack([
            student_probs.flatten(),
            mean_teacher_probs.flatten()
        ]))[0, 1].item()
        
        relationship["prob_correlation"] = correlation
        
        return relationship

    def export_analysis(
        self,
        results: Dict[str, Any],
        export_dir: str,
        formats: List[str] = ["json", "html", "pdf"]
    ) -> Dict[str, str]:
        """Export analysis results.
        
        Args:
            results: Analysis results
            export_dir: Export directory
            formats: Export formats
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export JSON
        if "json" in formats:
            json_path = export_dir / "analysis_results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            exported_files["json"] = str(json_path)
        
        # Export HTML
        if "html" in formats:
            html_path = export_dir / "analysis_report.html"
            self._export_html_report(results, html_path)
            exported_files["html"] = str(html_path)
        
        # Export PDF
        if "pdf" in formats:
            pdf_path = export_dir / "analysis_report.pdf"
            self._export_pdf_report(results, pdf_path)
            exported_files["pdf"] = str(pdf_path)
        
        return exported_files

    def _export_html_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Export analysis results as HTML report."""
        # Generate plots
        plots = self.plot_analysis(results)
        
        # Create HTML template
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Distillation Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; }
                .plot { margin: 20px 0; }
                .stats { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Distillation Analysis Report</h1>
            
            {% for plot_name, plot in plots.items() %}
            <div class="plot">
                <h2>{{ plot_name }}</h2>
                {{ plot }}
            </div>
            {% endfor %}
            
            <div class="stats">
                <h2>Statistics Summary</h2>
                <pre>{{ stats }}</pre>
            </div>
        </body>
        </html>
        """
        
        # Convert plots to HTML
        plot_html = {
            name: plot.to_html(full_html=False)
            for name, plot in plots.items()
        }
        
        # Generate statistics summary
        stats_summary = json.dumps(results, indent=2)
        
        # Render template
        env = jinja2.Environment()
        template = env.from_string(template)
        html = template.render(
            plots=plot_html,
            stats=stats_summary
        )
        
        # Save HTML
        with open(output_path, "w") as f:
            f.write(html)

    def _export_pdf_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Export analysis results as PDF report."""
        try:
            import pdfkit
        except ImportError:
            print("pdfkit not installed. Please install with: pip install pdfkit")
            return
        
        # First generate HTML
        html_path = output_path.with_suffix(".html")
        self._export_html_report(results, html_path)
        
        # Convert HTML to PDF
        pdfkit.from_file(str(html_path), str(output_path))
        
        # Clean up HTML
        html_path.unlink()

    def _analyze_learning_dynamics(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze learning dynamics during distillation."""
        dynamics = {}
        
        # Extract metrics over time
        metrics_over_time = {
            "student_loss": [],
            "teacher_loss": [],
            "distill_loss": [],
            "accuracy": [],
            "agreement": []
        }
        
        for step in history:
            for metric in metrics_over_time:
                if metric in step:
                    metrics_over_time[metric].append(step[metric])
        
        # Analyze convergence
        dynamics["convergence"] = {
            metric: {
                "final": float(values[-1]) if values else 0.0,
                "min": float(min(values)) if values else 0.0,
                "max": float(max(values)) if values else 0.0,
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
                "convergence_step": self._find_convergence_step(values)
            }
            for metric, values in metrics_over_time.items()
        }
        
        # Analyze learning phases
        dynamics["phases"] = self._analyze_learning_phases(metrics_over_time)
        
        # Analyze metric correlations
        dynamics["correlations"] = self._analyze_metric_correlations(metrics_over_time)
        
        return dynamics

    def _find_convergence_step(
        self,
        values: List[float],
        window: int = 10,
        threshold: float = 0.01
    ) -> Optional[int]:
        """Find step where metric converges."""
        if not values or len(values) < window:
            return None
        
        values = np.array(values)
        for i in range(len(values) - window):
            window_vals = values[i:i + window]
            if np.std(window_vals) < threshold:
                return i
            
        return None

    def _analyze_learning_phases(
        self,
        metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze different phases of learning."""
        phases = {}
        
        # Detect initial learning phase
        student_loss = np.array(metrics["student_loss"])
        initial_phase_end = np.argmin(student_loss[:len(student_loss)//3])
        
        # Detect plateau phase
        plateau_start = self._find_plateau_start(student_loss)
        
        # Detect fine-tuning phase
        if "agreement" in metrics:
            agreement = np.array(metrics["agreement"])
            finetuning_start = self._find_finetuning_start(student_loss, agreement)
        else:
            finetuning_start = None
        
        phases["boundaries"] = {
            "initial_phase_end": int(initial_phase_end),
            "plateau_start": int(plateau_start) if plateau_start is not None else None,
            "finetuning_start": int(finetuning_start) if finetuning_start is not None else None
        }
        
        # Analyze characteristics of each phase
        for metric, values in metrics.items():
            values = np.array(values)
            phases[metric] = {
                "initial": {
                    "mean": float(np.mean(values[:initial_phase_end])),
                    "slope": float(np.polyfit(
                        range(initial_phase_end),
                        values[:initial_phase_end],
                        deg=1
                    )[0])
                }
            }
            if plateau_start is not None:
                phases[metric]["plateau"] = {
                    "mean": float(np.mean(values[plateau_start:])),
                    "std": float(np.std(values[plateau_start:]))
                }
        
        return phases

    def _find_plateau_start(
        self,
        values: np.ndarray,
        window: int = 20,
        threshold: float = 0.005
    ) -> Optional[int]:
        """Find start of plateau phase."""
        if len(values) < window:
            return None
        
        slopes = []
        for i in range(len(values) - window):
            slope = np.polyfit(
                range(window),
                values[i:i + window],
                deg=1
            )[0]
            slopes.append(abs(slope))
        
        slopes = np.array(slopes)
        plateau_candidates = np.where(slopes < threshold)[0]
        
        if len(plateau_candidates) > 0:
            return plateau_candidates[0]
        return None

    def _find_finetuning_start(
        self,
        loss: np.ndarray,
        agreement: np.ndarray,
        window: int = 20
    ) -> Optional[int]:
        """Find start of fine-tuning phase."""
        if len(loss) < window or len(agreement) < window:
            return None
        
        # Look for point where loss stabilizes but agreement still improves
        loss_slopes = []
        agreement_slopes = []
        
        for i in range(len(loss) - window):
            loss_slope = np.polyfit(
                range(window),
                loss[i:i + window],
                deg=1
            )[0]
            loss_slopes.append(abs(loss_slope))
            
            agreement_slope = np.polyfit(
                range(window),
                agreement[i:i + window],
                deg=1
            )[0]
            agreement_slopes.append(agreement_slope)
        
        loss_slopes = np.array(loss_slopes)
        agreement_slopes = np.array(agreement_slopes)
        
        # Find points where loss is stable but agreement improves
        candidates = np.where(
            (loss_slopes < 0.01) & (agreement_slopes > 0.001)
        )[0]
        
        if len(candidates) > 0:
            return candidates[0]
        return None

    def _analyze_metric_correlations(
        self,
        metrics: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between different metrics."""
        correlations = {}
        
        metric_names = list(metrics.keys())
        for i, metric1 in enumerate(metric_names):
            correlations[metric1] = {}
            for metric2 in metric_names[i+1:]:
                if len(metrics[metric1]) == len(metrics[metric2]):
                    correlation = np.corrcoef(
                        metrics[metric1],
                        metrics[metric2]
                    )[0, 1]
                    correlations[metric1][metric2] = float(correlation)
        
        return correlations

    def _plot_learning_dynamics(
        self,
        dynamics: Dict[str, Any]
    ) -> go.Figure:
        """Plot learning dynamics analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Metric Convergence',
                'Learning Phases',
                'Phase Characteristics',
                'Metric Correlations'
            )
        )
        
        # Plot metric convergence
        metrics = dynamics["convergence"]
        for metric, stats in metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=[0, stats["convergence_step"]],
                    y=[stats["min"], stats["final"]],
                    name=f"{metric}_convergence",
                    mode="lines+markers"
                ),
                row=1, col=1
            )
        
        # Plot learning phases
        phases = dynamics["phases"]
        boundaries = phases["boundaries"]
        
        for boundary, step in boundaries.items():
            if step is not None:
                fig.add_vline(
                    x=step,
                    line_dash="dash",
                    annotation_text=boundary,
                    row=1, col=2
                )
        
        # Plot phase characteristics
        metrics = list(phases.keys())
        metrics.remove("boundaries")
        
        for metric in metrics:
            phase_stats = phases[metric]
            for phase, stats in phase_stats.items():
                fig.add_trace(
                    go.Bar(
                        x=[f"{metric}_{phase}"],
                        y=[stats["mean"]],
                        name=f"{metric}_{phase}"
                    ),
                    row=2, col=1
                )
        
        # Plot metric correlations
        correlations = dynamics["correlations"]
        correlation_matrix = []
        metric_names = []
        
        for metric1, corrs in correlations.items():
            metric_names.append(metric1)
            row = []
            for metric2 in metric_names:
                if metric2 == metric1:
                    row.append(1.0)
                elif metric2 in corrs:
                    row.append(corrs[metric2])
                else:
                    row.append(None)
            correlation_matrix.append(row)
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=metric_names,
                y=metric_names,
                colorscale="RdBu",
                zmin=-1,
                zmax=1
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Learning Dynamics Analysis",
            showlegend=True
        )
        
        return fig

    def _analyze_compression_efficiency(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze model compression efficiency."""
        compression = {}
        
        # Calculate model sizes
        student_size = sum(p.numel() for p in student_model.parameters())
        teacher_size = sum(p.numel() for p in teacher_model.parameters())
        
        compression["size"] = {
            "student": student_size,
            "teacher": teacher_size,
            "compression_ratio": float(teacher_size / student_size)
        }
        
        # Calculate FLOPs
        student_flops = self._calculate_flops(student_model)
        teacher_flops = self._calculate_flops(teacher_model)
        
        compression["compute"] = {
            "student_flops": student_flops,
            "teacher_flops": teacher_flops,
            "speedup_ratio": float(teacher_flops / student_flops)
        }
        
        # Calculate efficiency metrics
        compression["efficiency"] = {
            "size_performance_ratio": float(
                performance_metrics["accuracy"] / compression["size"]["compression_ratio"]
            ),
            "compute_performance_ratio": float(
                performance_metrics["accuracy"] / compression["compute"]["speedup_ratio"]
            )
        }
        
        return compression

    def _calculate_flops(
        self,
        model: torch.nn.Module
    ) -> int:
        """Calculate approximate FLOPs for a model."""
        total_flops = 0
        
        def conv_flops(layer, input_shape):
            batch, in_c, in_h, in_w = input_shape
            out_c = layer.out_channels
            k_h, k_w = layer.kernel_size
            return batch * out_c * in_h * in_w * k_h * k_w * in_c
        
        def linear_flops(layer, input_shape):
            batch = input_shape[0]
            return batch * layer.in_features * layer.out_features
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                total_flops += conv_flops(module, (1, *module.in_channels, 224, 224))
            elif isinstance(module, torch.nn.Linear):
                total_flops += linear_flops(module, (1, module.in_features))
                
        return total_flops

    def _plot_compression_analysis(
        self,
        compression: Dict[str, Any]
    ) -> go.Figure:
        """Plot compression analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Size Comparison',
                'Compute Requirements',
                'Efficiency Metrics',
                'Size-Performance Trade-off'
            )
        )
        
        # Plot model sizes
        sizes = [
            compression["size"]["student"],
            compression["size"]["teacher"]
        ]
        fig.add_trace(
            go.Bar(
                x=["Student", "Teacher"],
                y=sizes,
                name="Model Size"
            ),
            row=1, col=1
        )
        
        # Plot compute requirements
        flops = [
            compression["compute"]["student_flops"],
            compression["compute"]["teacher_flops"]
        ]
        fig.add_trace(
            go.Bar(
                x=["Student", "Teacher"],
                y=flops,
                name="FLOPs"
            ),
            row=1, col=2
        )
        
        # Plot efficiency metrics
        metrics = compression["efficiency"]
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                name="Efficiency"
            ),
            row=2, col=1
        )
        
        # Plot size-performance trade-off
        fig.add_trace(
            go.Scatter(
                x=[compression["size"]["compression_ratio"]],
                y=[compression["efficiency"]["size_performance_ratio"]],
                mode="markers",
                name="Trade-off",
                marker=dict(size=15)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title="Compression Analysis",
            showlegend=True
        )
        
        return fig

    def _analyze_generalization(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Analyze model generalization capabilities."""
        generalization = {}
        
        # Analyze prediction confidence
        student_probs = F.softmax(student_outputs["logits"], dim=1)
        teacher_probs = F.softmax(teacher_outputs["logits"], dim=1)
        
        generalization["confidence"] = {
            "student": {
                "mean": float(student_probs.max(dim=1)[0].mean()),
                "std": float(student_probs.max(dim=1)[0].std()),
                "entropy": float(-torch.sum(
                    student_probs * torch.log(student_probs + 1e-10),
                    dim=1
                ).mean())
            },
            "teacher": {
                "mean": float(teacher_probs.max(dim=1)[0].mean()),
                "std": float(teacher_probs.max(dim=1)[0].std()),
                "entropy": float(-torch.sum(
                    teacher_probs * torch.log(teacher_probs + 1e-10),
                    dim=1
                ).mean())
            }
        }
        
        # Analyze calibration
        student_calibration = self._compute_calibration(student_probs, labels)
        teacher_calibration = self._compute_calibration(teacher_probs, labels)
        
        generalization["calibration"] = {
            "student": student_calibration,
            "teacher": teacher_calibration
        }
        
        # Analyze out-of-distribution behavior if validation data is provided
        if validation_data is not None:
            val_data, val_labels = validation_data
            generalization["validation"] = self._analyze_ood_behavior(
                student_outputs, teacher_outputs,
                val_data, val_labels
            )
        
        return generalization

    def _compute_calibration(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration metrics."""
        confidences = probs.max(dim=1)[0]
        predictions = probs.argmax(dim=1)
        accuracies = (predictions == labels).float()
        
        # Expected Calibration Error
        ece = 0.0
        for i in range(n_bins):
            bin_start = i / n_bins
            bin_end = (i + 1) / n_bins
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)
            if bin_mask.any():
                bin_acc = accuracies[bin_mask].mean()
                bin_conf = confidences[bin_mask].mean()
                bin_size = bin_mask.sum() / len(confidences)
                ece += bin_size * abs(bin_acc - bin_conf)
        
        # Maximum Calibration Error
        mce = 0.0
        for i in range(n_bins):
            bin_start = i / n_bins
            bin_end = (i + 1) / n_bins
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)
            if bin_mask.any():
                bin_acc = accuracies[bin_mask].mean()
                bin_conf = confidences[bin_mask].mean()
                mce = max(mce, abs(bin_acc - bin_conf))
        
        return {
            "ece": float(ece),
            "mce": float(mce),
            "avg_confidence": float(confidences.mean()),
            "avg_accuracy": float(accuracies.mean())
        }

    def _analyze_ood_behavior(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        val_data: torch.Tensor,
        val_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze out-of-distribution behavior."""
        ood = {}
        
        # Get predictions on validation data
        with torch.no_grad():
            student_val = self.student(val_data)
            teacher_val = self.teacher(val_data)
        
        # Compute confidence on validation data
        student_val_probs = F.softmax(student_val["logits"], dim=1)
        teacher_val_probs = F.softmax(teacher_val["logits"], dim=1)
        
        # Analyze confidence shift
        ood["confidence_shift"] = {
            "student": float(
                student_val_probs.max(dim=1)[0].mean() -
                F.softmax(student_outputs["logits"], dim=1).max(dim=1)[0].mean()
            ),
            "teacher": float(
                teacher_val_probs.max(dim=1)[0].mean() -
                F.softmax(teacher_outputs["logits"], dim=1).max(dim=1)[0].mean()
            )
        }
        
        # Analyze prediction stability
        ood["prediction_stability"] = {
            "student": float((
                student_val_probs.argmax(dim=1) ==
                F.softmax(student_outputs["logits"], dim=1).argmax(dim=1)
            ).float().mean()),
            "teacher": float((
                teacher_val_probs.argmax(dim=1) ==
                F.softmax(teacher_outputs["logits"], dim=1).argmax(dim=1)
            ).float().mean())
        }
        
        # Analyze validation performance
        ood["validation_performance"] = {
            "student": {
                "accuracy": float((
                    student_val_probs.argmax(dim=1) == val_labels
                ).float().mean()),
                "confidence": float(student_val_probs.max(dim=1)[0].mean())
            },
            "teacher": {
                "accuracy": float((
                    teacher_val_probs.argmax(dim=1) == val_labels
                ).float().mean()),
                "confidence": float(teacher_val_probs.max(dim=1)[0].mean())
            }
        }
        
        return ood 