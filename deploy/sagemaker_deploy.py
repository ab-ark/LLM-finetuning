"""
AWS SageMaker deployment helpers for AbArk LLM Fine-Tuning Recipes.
Deploy fine-tuned models to SageMaker real-time inference endpoints.

Requires: boto3, sagemaker
Install: pip install boto3 sagemaker
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """
    Deploy a fine-tuned HuggingFace model to SageMaker.

    Usage:
        deployer = SageMakerDeployer(role_arn="arn:aws:iam::123456:role/SageMakerRole")
        endpoint = deployer.deploy(
            model_s3_uri="s3://my-bucket/models/lora_model/",
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            instance_type="ml.g5.2xlarge",
        )
        response = deployer.predict(endpoint, "What is machine learning?")
    """

    def __init__(
        self,
        role_arn: Optional[str] = None,
        region: str = "us-east-1",
    ):
        self.role_arn = role_arn or os.environ.get("SAGEMAKER_ROLE_ARN", "")
        self.region = region

    def upload_model(self, local_model_dir: str, s3_bucket: str, s3_prefix: str = "abark-models") -> str:
        """Upload local model directory to S3. Returns S3 URI."""
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self.region)
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")

        import os
        s3_uri_parts = []
        for root, dirs, files in os.walk(local_model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative = os.path.relpath(local_path, local_model_dir)
                s3_key = f"{s3_prefix}/{relative}"
                logger.info(f"Uploading {local_path} → s3://{s3_bucket}/{s3_key}")
                s3.upload_file(local_path, s3_bucket, s3_key)
                s3_uri_parts.append(s3_key)

        base_uri = f"s3://{s3_bucket}/{s3_prefix}/"
        logger.info(f"Model uploaded to: {base_uri}")
        return base_uri

    def deploy(
        self,
        model_s3_uri: str,
        model_id: str,
        endpoint_name: str = "abark-llm-endpoint",
        instance_type: str = "ml.g5.2xlarge",
        num_gpus: int = 1,
        max_input_length: int = 2048,
        max_total_tokens: int = 4096,
    ) -> str:
        """Deploy model to SageMaker. Returns endpoint name."""
        try:
            from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
            import sagemaker
        except ImportError:
            raise ImportError("Install sagemaker: pip install sagemaker")

        logger.info(f"Deploying to SageMaker endpoint: {endpoint_name}")
        logger.info(f"  Instance type: {instance_type}")
        logger.info(f"  Model S3 URI:  {model_s3_uri}")

        image_uri = get_huggingface_llm_image_uri("huggingface", version="2.0.2")
        env = {
            "HF_MODEL_ID": model_id,
            "SM_NUM_GPUS": str(num_gpus),
            "MAX_INPUT_LENGTH": str(max_input_length),
            "MAX_TOTAL_TOKENS": str(max_total_tokens),
        }

        hf_model = HuggingFaceModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            image_uri=image_uri,
            env=env,
        )
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        logger.info(f"Endpoint deployed: {endpoint_name}")
        return endpoint_name

    def predict(self, endpoint_name: str, prompt: str, max_new_tokens: int = 256) -> str:
        """Run inference on a deployed SageMaker endpoint."""
        try:
            import boto3
            runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7},
        }
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read())
        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return str(result)

    def delete_endpoint(self, endpoint_name: str):
        """Delete a SageMaker endpoint to stop billing."""
        try:
            import boto3
            sm = boto3.client("sagemaker", region_name=self.region)
            sm.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted: {endpoint_name}")
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
