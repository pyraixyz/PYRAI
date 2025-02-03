from typing import Dict, Any, Optional, List
import yaml
import docker
import kubernetes
from pathlib import Path
from datetime import datetime
from .config import DeploymentConfig
from ..core.versioning import VersionManager
from ..monitoring.logger import ModelLogger

class DeploymentManager:
    """Manages model deployments."""
    
    def __init__(self,
                 version_manager: VersionManager,
                 logger: ModelLogger,
                 config_path: str,
                 registry_url: str):
        self.version_manager = version_manager
        self.logger = logger
        self.config_path = Path(config_path)
        self.registry_url = registry_url
        
        # Initialize clients
        self.docker_client = docker.from_env()
        self.k8s_client = kubernetes.client.ApiClient()
        self.k8s_apps = kubernetes.client.AppsV1Api(self.k8s_client)
        self.k8s_core = kubernetes.client.CoreV1Api(self.k8s_client)
        
    def deploy_model(self, config: DeploymentConfig) -> str:
        """Deploy a model to production."""
        try:
            # Load model version
            model, version = self.version_manager.load_version(
                config.model_id,
                config.version_id
            )
            
            # Build and push container
            image_tag = self._build_container(model, version, config)
            
            # Deploy to Kubernetes
            deployment_name = self._deploy_to_kubernetes(image_tag, config)
            
            # Log deployment
            self.logger.info(
                "model_deployment",
                model_id=config.model_id,
                version_id=config.version_id,
                deployment_name=deployment_name,
                config=config.dict()
            )
            
            return deployment_name
            
        except Exception as e:
            self.logger.error(
                "deployment_error",
                error_type=type(e).__name__,
                error_message=str(e),
                config=config.dict()
            )
            raise
    
    def _build_container(self,
                        model: Any,
                        version: Any,
                        config: DeploymentConfig) -> str:
        """Build and push model container."""
        # Create temporary build directory
        build_dir = self.config_path / 'builds' / datetime.now().strftime('%Y%m%d_%H%M%S')
        build_dir.mkdir(parents=True)
        
        try:
            # Save model artifacts
            model_path = build_dir / 'model'
            self.version_manager._save_model(model, model_path)
            
            # Create Dockerfile
            dockerfile = self._generate_dockerfile(config)
            dockerfile_path = build_dir / 'Dockerfile'
            dockerfile_path.write_text(dockerfile)
            
            # Build image
            image_tag = f"{self.registry_url}/{config.model_id}:{config.version_id}"
            self.docker_client.images.build(
                path=str(build_dir),
                tag=image_tag,
                rm=True
            )
            
            # Push image
            self.docker_client.images.push(image_tag)
            
            return image_tag
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(build_dir)
    
    def _generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Generate Dockerfile for model deployment."""
        return f"""
FROM python:3.8-slim

# Install dependencies
RUN pip install pyrai torch tensorflow fastapi uvicorn

# Copy model artifacts
COPY model /app/model

# Copy server code
COPY server.py /app/

WORKDIR /app
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    def _deploy_to_kubernetes(self,
                            image_tag: str,
                            config: DeploymentConfig) -> str:
        """Deploy model to Kubernetes."""
        # Generate deployment name
        deployment_name = f"{config.name}-{config.version_id}".lower()
        
        # Create deployment
        deployment = kubernetes.client.V1Deployment(
            metadata=kubernetes.client.V1ObjectMeta(name=deployment_name),
            spec=kubernetes.client.V1DeploymentSpec(
                replicas=config.scaling['min_replicas'],
                selector=kubernetes.client.V1LabelSelector(
                    match_labels={"app": deployment_name}
                ),
                template=kubernetes.client.V1PodTemplateSpec(
                    metadata=kubernetes.client.V1ObjectMeta(
                        labels={"app": deployment_name}
                    ),
                    spec=kubernetes.client.V1PodSpec(
                        containers=[
                            kubernetes.client.V1Container(
                                name="model",
                                image=image_tag,
                                ports=[
                                    kubernetes.client.V1ContainerPort(
                                        container_port=8000
                                    )
                                ],
                                resources=kubernetes.client.V1ResourceRequirements(
                                    requests=config.resources,
                                    limits=config.resources
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        # Create service
        service = kubernetes.client.V1Service(
            metadata=kubernetes.client.V1ObjectMeta(name=deployment_name),
            spec=kubernetes.client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[
                    kubernetes.client.V1ServicePort(
                        port=80,
                        target_port=8000
                    )
                ]
            )
        )
        
        # Create HPA if autoscaling is enabled
        if config.scaling.get('max_replicas'):
            hpa = kubernetes.client.V2beta2HorizontalPodAutoscaler(
                metadata=kubernetes.client.V1ObjectMeta(name=deployment_name),
                spec=kubernetes.client.V2beta2HorizontalPodAutoscalerSpec(
                    scale_target_ref=kubernetes.client.V2beta2CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=config.scaling['min_replicas'],
                    max_replicas=config.scaling['max_replicas'],
                    metrics=[
                        kubernetes.client.V2beta2MetricSpec(
                            type="Resource",
                            resource=kubernetes.client.V2beta2ResourceMetricSource(
                                name="cpu",
                                target=kubernetes.client.V2beta2MetricTarget(
                                    type="Utilization",
                                    average_utilization=config.scaling['target_cpu_utilization']
                                )
                            )
                        )
                    ]
                )
            )
            
            # Apply HPA
            kubernetes.client.AutoscalingV2beta2Api(self.k8s_client).create_namespaced_horizontal_pod_autoscaler(
                namespace="default",
                body=hpa
            )
        
        # Apply deployment and service
        self.k8s_apps.create_namespaced_deployment(
            namespace="default",
            body=deployment
        )
        
        self.k8s_core.create_namespaced_service(
            namespace="default",
            body=service
        )
        
        return deployment_name
    
    def list_deployments(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all deployments."""
        deployments = self.k8s_apps.list_namespaced_deployment(
            namespace="default",
            label_selector=f"model_id={model_id}" if model_id else None
        )
        
        return [
            {
                'name': d.metadata.name,
                'model_id': d.metadata.labels.get('model_id'),
                'version_id': d.metadata.labels.get('version_id'),
                'replicas': d.spec.replicas,
                'available_replicas': d.status.available_replicas,
                'created_at': d.metadata.creation_timestamp
            }
            for d in deployments.items
        ]
    
    def delete_deployment(self, deployment_name: str) -> None:
        """Delete a deployment."""
        # Delete deployment
        self.k8s_apps.delete_namespaced_deployment(
            name=deployment_name,
            namespace="default"
        )
        
        # Delete service
        self.k8s_core.delete_namespaced_service(
            name=deployment_name,
            namespace="default"
        )
        
        # Delete HPA if exists
        try:
            kubernetes.client.AutoscalingV2beta2Api(self.k8s_client).delete_namespaced_horizontal_pod_autoscaler(
                name=deployment_name,
                namespace="default"
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:  # Not found
                raise 