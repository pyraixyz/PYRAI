from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import time
from dataclasses import dataclass
from enum import Enum
from .auto_responder import ResponseAction, AutoResponder
from .response_evaluator import ResponseEvaluation, ResponseOutcome
from .monitor import RuntimeStats
from .predictor import PredictionResult

class ControlMode(Enum):
    """Performance control modes."""
    AUTOMATIC = "automatic"
    SEMI_AUTO = "semi_auto"
    MANUAL = "manual"

@dataclass
class ControlDecision:
    """Control decision details."""
    action: ResponseAction
    params: Dict[str, Any]
    reason: str
    confidence: float
    requires_approval: bool

class InteractiveController:
    """Interactive performance control."""
    
    def __init__(
        self,
        responder: AutoResponder,
        mode: ControlMode = ControlMode.SEMI_AUTO,
        approval_threshold: float = 0.9,
        approval_callback: Optional[Callable[[ControlDecision], bool]] = None
    ):
        """Initialize controller.
        
        Args:
            responder: Automatic responder
            mode: Control mode
            approval_threshold: Confidence threshold for auto-approval
            approval_callback: Optional callback for manual approval
        """
        self.responder = responder
        self.mode = mode
        self.approval_threshold = approval_threshold
        self.approval_callback = approval_callback
        self.pending_decisions: List[ControlDecision] = []
        self.approved_decisions: List[ControlDecision] = []
        self.rejected_decisions: List[ControlDecision] = []
        
    def update(
        self,
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult],
        stats: RuntimeStats,
        evaluations: List[ResponseEvaluation]
    ) -> Optional[Tuple[ResponseAction, Dict[str, Any]]]:
        """Update controller with new data.
        
        Args:
            alerts: Active alerts
            predictions: Performance predictions
            stats: Current statistics
            evaluations: Response evaluations
            
        Returns:
            Response action and parameters if approved
        """
        # Generate response decision
        response = self.responder.respond(alerts, predictions, stats)
        if not response:
            return None
            
        action, params = response
        
        # Create control decision
        decision = self._create_decision(
            action,
            params,
            alerts,
            predictions,
            evaluations
        )
        
        # Handle decision based on mode
        if self.mode == ControlMode.AUTOMATIC:
            return self._handle_automatic(decision)
        elif self.mode == ControlMode.SEMI_AUTO:
            return self._handle_semi_auto(decision)
        else:  # MANUAL
            return self._handle_manual(decision)
            
    def _create_decision(
        self,
        action: ResponseAction,
        params: Dict[str, Any],
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult],
        evaluations: List[ResponseEvaluation]
    ) -> ControlDecision:
        """Create control decision."""
        # Calculate confidence based on predictions
        prediction_confidence = np.mean([
            pred.confidence
            for pred in predictions.values()
        ])
        
        # Consider historical success
        similar_evals = [
            eval for eval in evaluations
            if eval.action == action
        ]
        
        if similar_evals:
            success_rate = sum(
                1 for eval in similar_evals
                if eval.outcome in [ResponseOutcome.SUCCESS, ResponseOutcome.PARTIAL]
            ) / len(similar_evals)
        else:
            success_rate = 0.5
            
        # Combine confidences
        confidence = 0.7 * prediction_confidence + 0.3 * success_rate
        
        # Determine if approval needed
        requires_approval = (
            confidence < self.approval_threshold or
            self.mode != ControlMode.AUTOMATIC
        )
        
        return ControlDecision(
            action=action,
            params=params,
            reason=params.get("reason", "unknown"),
            confidence=confidence,
            requires_approval=requires_approval
        )
        
    def _handle_automatic(
        self,
        decision: ControlDecision
    ) -> Optional[Tuple[ResponseAction, Dict[str, Any]]]:
        """Handle automatic mode decision."""
        if decision.confidence >= self.approval_threshold:
            self.approved_decisions.append(decision)
            return decision.action, decision.params
            
        self.pending_decisions.append(decision)
        return None
        
    def _handle_semi_auto(
        self,
        decision: ControlDecision
    ) -> Optional[Tuple[ResponseAction, Dict[str, Any]]]:
        """Handle semi-automatic mode decision."""
        if decision.confidence >= self.approval_threshold:
            self.approved_decisions.append(decision)
            return decision.action, decision.params
            
        if self.approval_callback:
            if self.approval_callback(decision):
                self.approved_decisions.append(decision)
                return decision.action, decision.params
            else:
                self.rejected_decisions.append(decision)
                
        self.pending_decisions.append(decision)
        return None
        
    def _handle_manual(
        self,
        decision: ControlDecision
    ) -> Optional[Tuple[ResponseAction, Dict[str, Any]]]:
        """Handle manual mode decision."""
        if self.approval_callback and self.approval_callback(decision):
            self.approved_decisions.append(decision)
            return decision.action, decision.params
        else:
            self.pending_decisions.append(decision)
            
        return None
        
    def get_pending_decisions(self) -> List[ControlDecision]:
        """Get pending decisions requiring approval."""
        return self.pending_decisions
        
    def approve_decision(
        self,
        decision: ControlDecision
    ) -> Optional[Tuple[ResponseAction, Dict[str, Any]]]:
        """Approve pending decision."""
        if decision in self.pending_decisions:
            self.pending_decisions.remove(decision)
            self.approved_decisions.append(decision)
            return decision.action, decision.params
            
        return None
        
    def reject_decision(self, decision: ControlDecision):
        """Reject pending decision."""
        if decision in self.pending_decisions:
            self.pending_decisions.remove(decision)
            self.rejected_decisions.append(decision) 