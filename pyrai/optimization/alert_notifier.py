from typing import Any, Dict, List, Optional, Union, Tuple
import smtplib
import requests
from email.mime.text import MIMEText
from dataclasses import dataclass
from enum import Enum
from .alert_rules import Alert, AlertLevel, AlertState

class NotificationType(Enum):
    """Alert notification types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"

@dataclass
class NotificationConfig:
    """Notification configuration."""
    type: NotificationType
    settings: Dict[str, str]
    min_level: AlertLevel = AlertLevel.WARNING
    template: Optional[str] = None

class AlertNotifier:
    """Manages alert notifications."""
    
    def __init__(
        self,
        configs: List[NotificationConfig]
    ):
        """Initialize notifier.
        
        Args:
            configs: Notification configurations
        """
        self.configs = configs
        self.sent_alerts: Dict[str, List[float]] = {}  # alert_id -> timestamps
        
    def notify(self, alert: Alert):
        """Send alert notification."""
        if alert.state not in [AlertState.FIRING, AlertState.RESOLVED]:
            return
            
        alert_id = f"{alert.rule.name}_{alert.start_time}"
        
        for config in self.configs:
            if alert.rule.level.value < config.min_level.value:
                continue
                
            if config.type == NotificationType.EMAIL:
                self._send_email(alert, config)
            elif config.type == NotificationType.SLACK:
                self._send_slack(alert, config)
            else:  # WEBHOOK
                self._send_webhook(alert, config)
                
            if alert_id not in self.sent_alerts:
                self.sent_alerts[alert_id] = []
            self.sent_alerts[alert_id].append(time.time())
            
    def _send_email(self, alert: Alert, config: NotificationConfig):
        """Send email notification."""
        smtp_server = config.settings.get("smtp_server")
        smtp_port = int(config.settings.get("smtp_port", "587"))
        sender = config.settings.get("sender")
        password = config.settings.get("password")
        recipients = config.settings.get("recipients", "").split(",")
        
        if not all([smtp_server, sender, password, recipients]):
            return
            
        subject = f"[{alert.rule.level.value.upper()}] {alert.rule.name}"
        body = self._format_message(alert, config.template)
        
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email: {e}")
            
    def _send_slack(self, alert: Alert, config: NotificationConfig):
        """Send Slack notification."""
        webhook_url = config.settings.get("webhook_url")
        if not webhook_url:
            return
            
        message = self._format_message(alert, config.template)
        color = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ffcc00",
            AlertLevel.CRITICAL: "#ff0000"
        }.get(alert.rule.level, "#cccccc")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"[{alert.rule.level.value.upper()}] {alert.rule.name}",
                "text": message,
                "fields": [
                    {
                        "title": "Value",
                        "value": f"{alert.value:.2f}",
                        "short": True
                    },
                    {
                        "title": "State",
                        "value": alert.state.value,
                        "short": True
                    }
                ]
            }]
        }
        
        try:
            requests.post(webhook_url, json=payload)
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            
    def _send_webhook(self, alert: Alert, config: NotificationConfig):
        """Send webhook notification."""
        webhook_url = config.settings.get("webhook_url")
        if not webhook_url:
            return
            
        payload = {
            "alert": {
                "name": alert.rule.name,
                "level": alert.rule.level.value,
                "state": alert.state.value,
                "value": alert.value,
                "description": alert.rule.description,
                "labels": alert.rule.labels,
                "annotations": alert.annotations,
                "start_time": alert.start_time,
                "last_update": alert.last_update
            }
        }
        
        try:
            requests.post(webhook_url, json=payload)
        except Exception as e:
            print(f"Failed to send webhook notification: {e}")
            
    def _format_message(
        self,
        alert: Alert,
        template: Optional[str]
    ) -> str:
        """Format alert message."""
        if template:
            return template.format(
                name=alert.rule.name,
                level=alert.rule.level.value,
                state=alert.state.value,
                value=alert.value,
                description=alert.rule.description
            )
            
        return (
            f"Alert: {alert.rule.name}\n"
            f"Level: {alert.rule.level.value}\n"
            f"State: {alert.state.value}\n"
            f"Value: {alert.value:.2f}\n"
            f"Description: {alert.rule.description}"
        ) 