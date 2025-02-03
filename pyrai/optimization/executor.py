from typing import Any, Dict, List, Optional, Union, Callable
import torch
import tensorflow as tf
from .advisor import OptimizationAdvisor, OptimizationAdvice
from .optimizer import TorchOptimizer, TensorFlowOptimizer
from .compression import ModelCompressor
from .distillation import DistillationLoss

class OptimizationExecutor:
    """Executes optimization suggestions automatically."""
    
    def __init__(
        self,
        advisor: OptimizationAdvisor,
        eval_fn: Optional[Callable] = None,
        target_metric: str = "latency",
        max_trials: int = 5
    ):
        """Initialize executor.
        
        Args:
            advisor: Optimization advisor
            eval_fn: Evaluation function
            target_metric: Metric to optimize
            max_trials: Maximum optimization trials
        """
        self.advisor = advisor
        self.eval_fn = eval_fn
        self.target_metric = target_metric
        self.max_trials = max_trials
        
        # Initialize optimizers
        self.torch_optimizer = TorchOptimizer()
        self.tf_optimizer = TensorFlowOptimizer()
        self.compressor = ModelCompressor({})
        
    def optimize(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        input_shape: Union[tuple, List[tuple]],
        teacher_model: Optional[Union[torch.nn.Module, tf.keras.Model]] = None
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Execute optimization pipeline.
        
        Args:
            model: Model to optimize
            input_shape: Input shape(s)
            teacher_model: Optional teacher model for distillation
            
        Returns:
            Optimized model
        """
        best_model = model
        best_metric = float('inf')
        
        for trial in range(self.max_trials):
            # Get optimization suggestions
            advice_list = self.advisor.analyze(model, input_shape)
            
            # Apply optimizations
            optimized_model = self._apply_optimizations(
                model,
                advice_list,
                teacher_model
            )
            
            # Evaluate
            if self.eval_fn is not None:
                metric = self.eval_fn(optimized_model)
                if metric < best_metric:
                    best_model = optimized_model
                    best_metric = metric
            else:
                best_model = optimized_model
                
        return best_model
        
    def _apply_optimizations(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        advice_list: List[OptimizationAdvice],
        teacher_model: Optional[Union[torch.nn.Module, tf.keras.Model]]
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Apply optimization suggestions."""
        optimized_model = model
        
        for advice in advice_list:
            if advice.target == "latency":
                # Apply quantization
                config = {"quantize": True, "quantization": {"method": "int8"}}
                if isinstance(model, torch.nn.Module):
                    optimized_model = self.torch_optimizer.optimize(
                        optimized_model,
                        config
                    )
                else:
                    optimized_model = self.tf_optimizer.optimize(
                        optimized_model,
                        config
                    )
                    
            elif advice.target == "memory" or advice.target == "efficiency":
                # Apply pruning
                config = {
                    "prune": True,
                    "pruning": {"sparsity": 0.5, "method": "magnitude"}
                }
                if isinstance(model, torch.nn.Module):
                    optimized_model = self.torch_optimizer.optimize(
                        optimized_model,
                        config
                    )
                else:
                    optimized_model = self.tf_optimizer.optimize(
                        optimized_model,
                        config
                    )
                    
            elif advice.target == "model_size" and teacher_model is not None:
                # Apply knowledge distillation
                optimized_model = self._apply_distillation(
                    optimized_model,
                    teacher_model
                )
                
        return optimized_model
        
    def _apply_distillation(
        self,
        student_model: Union[torch.nn.Module, tf.keras.Model],
        teacher_model: Union[torch.nn.Module, tf.keras.Model],
        epochs: int = 10,
        batch_size: int = 32,
        dataset: Optional[Any] = None
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Apply knowledge distillation.
        
        Args:
            student_model: Student model
            teacher_model: Teacher model
            epochs: Number of training epochs
            batch_size: Training batch size
            dataset: Training dataset
            
        Returns:
            Trained student model
        """
        if dataset is None:
            raise ValueError("Dataset is required for distillation training")
            
        # Initialize distillation loss
        distill_loss = DistillationLoss()
        
        if isinstance(student_model, torch.nn.Module):
            return self._train_torch_model(
                student_model,
                teacher_model,
                distill_loss,
                dataset,
                epochs,
                batch_size
            )
        else:
            return self._train_tf_model(
                student_model,
                teacher_model,
                distill_loss,
                dataset,
                epochs,
                batch_size
            )
            
    def _train_torch_model(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distill_loss: DistillationLoss,
        dataset: Any,
        epochs: int,
        batch_size: int
    ) -> torch.nn.Module:
        """Train PyTorch student model."""
        device = next(student_model.parameters()).device
        optimizer = torch.optim.Adam(student_model.parameters())
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        student_model.train()
        teacher_model.eval()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                student_logits = student_model(x)
                with torch.no_grad():
                    teacher_logits = teacher_model(x)
                    
                loss = distill_loss(student_logits, teacher_logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        return student_model
        
    def _train_tf_model(
        self,
        student_model: tf.keras.Model,
        teacher_model: tf.keras.Model,
        distill_loss: DistillationLoss,
        dataset: Any,
        epochs: int,
        batch_size: int
    ) -> tf.keras.Model:
        """Train TensorFlow student model."""
        optimizer = tf.keras.optimizers.Adam()
        
        # Convert dataset to tf.data.Dataset if needed
        if not isinstance(dataset, tf.data.Dataset):
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(1000).batch(batch_size)
        
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                student_logits = student_model(x, training=True)
                teacher_logits = teacher_model(x, training=False)
                loss = distill_loss(student_logits, teacher_logits, y)
                
            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
            return loss
            
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x, y in dataset:
                loss = train_step(x, y)
                total_loss += loss
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        return student_model 