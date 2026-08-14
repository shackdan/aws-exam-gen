"""
download_docs.py
────────────────
AWS Certification Document Downloader

Downloads official AWS exam guides, blueprints, whitepapers,
and service FAQs for any supported certification directly from
AWS public URLs into the correct data/ subdirectory for ingestion.

Usage:
  python download_docs.py --cert SAA-C03
  python download_docs.py --cert all
  python download_docs.py --cert SAA-C03 --dry-run
  python download_docs.py --cert SAA-C03 --force
  python download_docs.py --list

Document sources:
  - AWS Certification exam guide PDFs (official AWS training site)
  - AWS Whitepapers relevant to each certification domain
  - AWS Service FAQ pages (rendered to PDF via headless fetch)
  - AWS Well-Architected Framework pillars (where applicable)

All downloads are:
  - Verified via SHA-256 checksum where known
  - Skipped if the file already exists and --force is not set
  - Retried up to 3 times on transient network errors
  - Logged with file size and download duration


prep env:
pip install requests urllib3 click rich pydantic
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import click
import requests
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────
# Bootstrap project root onto sys.path
# ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_DIR, REGISTRY_PATH
from utils import load_registry, setup_logging

console = Console()
log     = logging.getLogger("aws_exam_gen.downloader")


# ─────────────────────────────────────────────
# Document catalogue
# Each entry:
#   url      – direct PDF download link
#   filename – saved filename under data/{cert_code}/
#   sha256   – expected checksum (None = skip verification)
#   category – exam_guide | whitepaper | faq | framework
#   required – True = abort if download fails; False = warn and continue
# ─────────────────────────────────────────────

DOCUMENT_CATALOGUE: Dict[str, List[Dict]] = {

    # ══════════════════════════════════════════
    # CLF-C02 — AWS Certified Cloud Practitioner
    # ══════════════════════════════════════════
    "CLF-C02": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-cloud-practitioner/AWS-Certified-Cloud-Practitioner_Exam-Guide.pdf",
            "filename": "CLF-C02_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/aws-overview.pdf",
            "filename": "AWS-Overview-Whitepaper.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/how-aws-pricing-works/how-aws-pricing-works.pdf",
            "filename": "AWS-Pricing-Overview.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/introduction-aws-security/introduction-aws-security.pdf",
            "filename": "Introduction-to-AWS-Security.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── Core service FAQs ──────────────────────
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rds/faqs/",
            "filename": "FAQ-RDS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/dynamodb/faqs/",
            "filename": "FAQ-DynamoDB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudwatch/faqs/",
            "filename": "FAQ-CloudWatch.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/sqs/faqs/",
            "filename": "FAQ-SQS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/sns/faqs/",
            "filename": "FAQ-SNS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudfront/faqs/",
            "filename": "FAQ-CloudFront.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/route53/faqs/",
            "filename": "FAQ-Route53.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # AIF-C01 — AWS Certified AI Practitioner
    # ══════════════════════════════════════════
    "AIF-C01": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf",
            "filename": "AIF-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-caf-for-ai/aws-caf-for-ai.pdf",
            "filename": "AWS-CAF-for-AI.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/strategy-enterprise-ready-gen-ai-platform/strategy-enterprise-ready-gen-ai-platform.pdf",
            "filename": "Enterprise-Ready-GenAI-Platform.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/ml-best-practices-healthcare-life-sciences/ml-best-practices-healthcare-life-sciences.pdf",
            "filename": "ML-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── AI/ML service FAQs ─────────────────────
        {
            "url"     : "https://aws.amazon.com/sagemaker/faqs/",
            "filename": "FAQ-SageMaker.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/bedrock/faqs/",
            "filename": "FAQ-Bedrock.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rekognition/faqs/",
            "filename": "FAQ-Rekognition.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/comprehend/faqs/",
            "filename": "FAQ-Comprehend.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/textract/faqs/",
            "filename": "FAQ-Textract.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/translate/faqs/",
            "filename": "FAQ-Translate.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/transcribe/faqs/",
            "filename": "FAQ-Transcribe.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/polly/faqs/",
            "filename": "FAQ-Polly.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lex/faqs/",
            "filename": "FAQ-Lex.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/personalize/faqs/",
            "filename": "FAQ-Personalize.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # SAA-C03 — AWS Certified Solutions Architect Associate
    # ══════════════════════════════════════════
    "SAA-C03": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-sa-assoc/AWS-Certified-Solutions-Architect-Associate_Exam-Guide.pdf",
            "filename": "SAA-C03_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/architecting-hipaa-security-and-compliance-on-aws/architecting-hipaa-security-and-compliance-on-aws.pdf",
            "filename": "Architecting-HIPAA-Security-Compliance.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.pdf",
            "filename": "Disaster-Recovery-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/whitepapers/latest/building-data-lakes/building-data-lakes.pdf",
            "filename": "Building-Data-Lakes-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/microservices-on-aws/microservices-on-aws.pdf",
            "filename": "Microservices-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/serverless-architectures-lambda/serverless-architectures-aws-lambda.pdf",
            "filename": "Serverless-Architectures-Lambda.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/wellarchitected-reliability-pillar.pdf",
            "filename": "Well-Architected-Reliability-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/wellarchitected-cost-optimization-pillar.pdf",
            "filename": "Well-Architected-Cost-Optimization-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/performance-efficiency-pillar/wellarchitected-performance-efficiency-pillar.pdf",
            "filename": "Well-Architected-Performance-Efficiency-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── In-scope service FAQs ──────────────────
        # Compute
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/autoscaling/faqs/",
            "filename": "FAQ-AutoScaling.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticloadbalancing/faqs/",
            "filename": "FAQ-ELB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Storage
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/efs/faq/",
            "filename": "FAQ-EFS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ebs/faqs/",
            "filename": "FAQ-EBS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Database
        {
            "url"     : "https://aws.amazon.com/rds/faqs/",
            "filename": "FAQ-RDS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rds/aurora/faqs/",
            "filename": "FAQ-Aurora.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/dynamodb/faqs/",
            "filename": "FAQ-DynamoDB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticache/faqs/",
            "filename": "FAQ-ElastiCache.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Networking & Content Delivery
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudfront/faqs/",
            "filename": "FAQ-CloudFront.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/route53/faqs/",
            "filename": "FAQ-Route53.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Application Integration
        {
            "url"     : "https://aws.amazon.com/sqs/faqs/",
            "filename": "FAQ-SQS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/sns/faqs/",
            "filename": "FAQ-SNS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Security & IAM
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Monitoring & Management
        {
            "url"     : "https://aws.amazon.com/cloudwatch/faqs/",
            "filename": "FAQ-CloudWatch.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # DVA-C02 — AWS Certified Developer Associate
    # ══════════════════════════════════════════
    "DVA-C02": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-dev-associate/AWS-Certified-Developer-Associate_Exam-Guide.pdf",
            "filename": "DVA-C02_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/serverless-architectures-lambda/serverless-architectures-aws-lambda.pdf",
            "filename": "Serverless-Architectures-Lambda.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/microservices-on-aws/microservices-on-aws.pdf",
            "filename": "Microservices-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/practicing-continuous-integration-continuous-delivery/practicing-continuous-integration-continuous-delivery.pdf",
            "filename": "CI-CD-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── Developer service FAQs ─────────────────
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/dynamodb/faqs/",
            "filename": "FAQ-DynamoDB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/api-gateway/faqs/",
            "filename": "FAQ-APIGateway.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/sqs/faqs/",
            "filename": "FAQ-SQS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/sns/faqs/",
            "filename": "FAQ-SNS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cognito/faqs/",
            "filename": "FAQ-Cognito.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticache/faqs/",
            "filename": "FAQ-ElastiCache.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/codedeploy/faqs/",
            "filename": "FAQ-CodeDeploy.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/codepipeline/faqs/",
            "filename": "FAQ-CodePipeline.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/codebuild/faqs/",
            "filename": "FAQ-CodeBuild.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/xray/faqs/",
            "filename": "FAQ-XRay.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticbeanstalk/faqs/",
            "filename": "FAQ-ElasticBeanstalk.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ecs/faqs/",
            "filename": "FAQ-ECS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kinesis/faqs/",
            "filename": "FAQ-Kinesis.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/step-functions/faqs/",
            "filename": "FAQ-StepFunctions.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/eventbridge/faqs/",
            "filename": "FAQ-EventBridge.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/secrets-manager/faqs/",
            "filename": "FAQ-SecretsManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kms/faqs/",
            "filename": "FAQ-KMS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudformation/faqs/",
            "filename": "FAQ-CloudFormation.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # SOA-C02 — AWS Certified SysOps Administrator Associate
    # ══════════════════════════════════════════
    "SOA-C02": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-sysops-associate/AWS-Certified-SysOps-Administrator-Associate_Exam-Guide.pdf",
            "filename": "SOA-C02_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/wellarchitected-reliability-pillar.pdf",
            "filename": "Well-Architected-Reliability-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.pdf",
            "filename": "Disaster-Recovery-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/introduction-aws-security/introduction-aws-security.pdf",
            "filename": "Introduction-to-AWS-Security.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── SysOps service FAQs ────────────────────
        {
            "url"     : "https://aws.amazon.com/cloudwatch/faqs/",
            "filename": "FAQ-CloudWatch.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudformation/faqs/",
            "filename": "FAQ-CloudFormation.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/systems-manager/faqs/",
            "filename": "FAQ-SystemsManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/autoscaling/faqs/",
            "filename": "FAQ-AutoScaling.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rds/faqs/",
            "filename": "FAQ-RDS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticloadbalancing/faqs/",
            "filename": "FAQ-ELB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudtrail/faqs/",
            "filename": "FAQ-CloudTrail.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/config/faqs/",
            "filename": "FAQ-Config.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/route53/faqs/",
            "filename": "FAQ-Route53.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudfront/faqs/",
            "filename": "FAQ-CloudFront.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # DEA-C01 — AWS Certified Data Engineer Associate
    # ══════════════════════════════════════════
    "DEA-C01": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-data-engineer-associate/AWS-Certified-Data-Engineer-Associate_Exam-Guide.pdf",
            "filename": "DEA-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/big-data-analytics-options/big-data-analytics-options.pdf",
            "filename": "Big-Data-Analytics-Options.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/building-data-lakes/building-data-lakes.pdf",
            "filename": "Building-Data-Lakes-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-glue-best-practices-build-performant-data-pipeline/aws-glue-best-practices-build-performant-data-pipeline.pdf",
            "filename": "AWS-Glue-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── Data engineering service FAQs ──────────
        {
            "url"     : "https://aws.amazon.com/kinesis/faqs/",
            "filename": "FAQ-Kinesis.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/glue/faqs/",
            "filename": "FAQ-Glue.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/redshift/faqs/",
            "filename": "FAQ-Redshift.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/dynamodb/faqs/",
            "filename": "FAQ-DynamoDB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rds/faqs/",
            "filename": "FAQ-RDS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/emr/faqs/",
            "filename": "FAQ-EMR.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/athena/faqs/",
            "filename": "FAQ-Athena.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/step-functions/faqs/",
            "filename": "FAQ-StepFunctions.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/eventbridge/faqs/",
            "filename": "FAQ-EventBridge.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/msk/faqs/",
            "filename": "FAQ-MSK.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lake-formation/faqs/",
            "filename": "FAQ-LakeFormation.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # MLA-C01 — AWS Certified Machine Learning Engineer Associate
    # ══════════════════════════════════════════
    "MLA-C01": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-machine-learning-engineer-associate/AWS-Certified-Machine-Learning-Engineer-Associate_Exam-Guide.pdf",
            "filename": "MLA-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-caf-for-ai/aws-caf-for-ai.pdf",
            "filename": "AWS-CAF-for-AI.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/strategy-enterprise-ready-gen-ai-platform/strategy-enterprise-ready-gen-ai-platform.pdf",
            "filename": "Enterprise-Ready-GenAI-Platform.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/ml-best-practices-healthcare-life-sciences/ml-best-practices-healthcare-life-sciences.pdf",
            "filename": "ML-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── ML engineering service FAQs ────────────
        {
            "url"     : "https://aws.amazon.com/sagemaker/faqs/",
            "filename": "FAQ-SageMaker.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/bedrock/faqs/",
            "filename": "FAQ-Bedrock.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/step-functions/faqs/",
            "filename": "FAQ-StepFunctions.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/glue/faqs/",
            "filename": "FAQ-Glue.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kinesis/faqs/",
            "filename": "FAQ-Kinesis.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/emr/faqs/",
            "filename": "FAQ-EMR.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rekognition/faqs/",
            "filename": "FAQ-Rekognition.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/comprehend/faqs/",
            "filename": "FAQ-Comprehend.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # SAP-C02 — AWS Certified Solutions Architect Professional
    # ══════════════════════════════════════════
    "SAP-C02": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-sa-pro/AWS-Certified-Solutions-Architect-Professional_Exam-Guide.pdf",
            "filename": "SAP-C02_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/wellarchitected-reliability-pillar.pdf",
            "filename": "Well-Architected-Reliability-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/wellarchitected-cost-optimization-pillar.pdf",
            "filename": "Well-Architected-Cost-Optimization-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/performance-efficiency-pillar/wellarchitected-performance-efficiency-pillar.pdf",
            "filename": "Well-Architected-Performance-Efficiency-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.pdf",
            "filename": "Disaster-Recovery-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/microservices-on-aws/microservices-on-aws.pdf",
            "filename": "Microservices-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/building-data-lakes/building-data-lakes.pdf",
            "filename": "Building-Data-Lakes-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── In-scope service FAQs (SAA-C03 superset) ──
        # Compute
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/autoscaling/faqs/",
            "filename": "FAQ-AutoScaling.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticloadbalancing/faqs/",
            "filename": "FAQ-ELB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ecs/faqs/",
            "filename": "FAQ-ECS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/eks/faqs/",
            "filename": "FAQ-EKS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Storage
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/efs/faq/",
            "filename": "FAQ-EFS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ebs/faqs/",
            "filename": "FAQ-EBS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Database
        {
            "url"     : "https://aws.amazon.com/rds/faqs/",
            "filename": "FAQ-RDS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rds/aurora/faqs/",
            "filename": "FAQ-Aurora.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/dynamodb/faqs/",
            "filename": "FAQ-DynamoDB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticache/faqs/",
            "filename": "FAQ-ElastiCache.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Networking
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudfront/faqs/",
            "filename": "FAQ-CloudFront.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/route53/faqs/",
            "filename": "FAQ-Route53.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/directconnect/faqs/",
            "filename": "FAQ-DirectConnect.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/transit-gateway/faqs/",
            "filename": "FAQ-TransitGateway.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Application integration
        {
            "url"     : "https://aws.amazon.com/sqs/faqs/",
            "filename": "FAQ-SQS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/sns/faqs/",
            "filename": "FAQ-SNS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/step-functions/faqs/",
            "filename": "FAQ-StepFunctions.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/eventbridge/faqs/",
            "filename": "FAQ-EventBridge.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        # Security, governance & management
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kms/faqs/",
            "filename": "FAQ-KMS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudwatch/faqs/",
            "filename": "FAQ-CloudWatch.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudformation/faqs/",
            "filename": "FAQ-CloudFormation.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/systems-manager/faqs/",
            "filename": "FAQ-SystemsManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/config/faqs/",
            "filename": "FAQ-Config.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudtrail/faqs/",
            "filename": "FAQ-CloudTrail.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/secrets-manager/faqs/",
            "filename": "FAQ-SecretsManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/organizations/faqs/",
            "filename": "FAQ-Organizations.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/waf/faqs/",
            "filename": "FAQ-WAF.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/shield/faqs/",
            "filename": "FAQ-Shield.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/guardduty/faqs/",
            "filename": "FAQ-GuardDuty.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/security-hub/faqs/",
            "filename": "FAQ-SecurityHub.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # DOP-C02 — AWS Certified DevOps Engineer Professional
    # ══════════════════════════════════════════
    "DOP-C02": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-devops-pro/AWS-Certified-DevOps-Engineer-Professional_Exam-Guide.pdf",
            "filename": "DOP-C02_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/practicing-continuous-integration-continuous-delivery/practicing-continuous-integration-continuous-delivery.pdf",
            "filename": "CI-CD-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/introduction-devops-aws/introduction-devops-aws.pdf",
            "filename": "Introduction-to-DevOps-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/wellarchitected-reliability-pillar.pdf",
            "filename": "Well-Architected-Reliability-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.pdf",
            "filename": "Disaster-Recovery-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── DevOps service FAQs ────────────────────
        {
            "url"     : "https://aws.amazon.com/codepipeline/faqs/",
            "filename": "FAQ-CodePipeline.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/codebuild/faqs/",
            "filename": "FAQ-CodeBuild.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/codedeploy/faqs/",
            "filename": "FAQ-CodeDeploy.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/codecommit/faqs/",
            "filename": "FAQ-CodeCommit.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudformation/faqs/",
            "filename": "FAQ-CloudFormation.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/systems-manager/faqs/",
            "filename": "FAQ-SystemsManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudwatch/faqs/",
            "filename": "FAQ-CloudWatch.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/config/faqs/",
            "filename": "FAQ-Config.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudtrail/faqs/",
            "filename": "FAQ-CloudTrail.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lambda/faqs/",
            "filename": "FAQ-Lambda.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ecs/faqs/",
            "filename": "FAQ-ECS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/eks/faqs/",
            "filename": "FAQ-EKS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticbeanstalk/faqs/",
            "filename": "FAQ-ElasticBeanstalk.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/autoscaling/faqs/",
            "filename": "FAQ-AutoScaling.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticloadbalancing/faqs/",
            "filename": "FAQ-ELB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/step-functions/faqs/",
            "filename": "FAQ-StepFunctions.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/eventbridge/faqs/",
            "filename": "FAQ-EventBridge.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kinesis/faqs/",
            "filename": "FAQ-Kinesis.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # SCS-C02 — AWS Certified Security Specialty
    # ══════════════════════════════════════════
    "SCS-C02": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-security-spec/AWS-Certified-Security-Specialty_Exam-Guide.pdf",
            "filename": "SCS-C02_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/introduction-aws-security/introduction-aws-security.pdf",
            "filename": "Introduction-to-AWS-Security.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/aws-security-incident-response-guide.pdf",
            "filename": "AWS-Security-Incident-Response-Guide.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/logical-separation/logical-separation.pdf",
            "filename": "AWS-Logical-Separation.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/kms/latest/developerguide/kms-dg.pdf",
            "filename": "KMS-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-best-practices-ddos-resiliency/aws-best-practices-ddos-resiliency.pdf",
            "filename": "AWS-DDoS-Resiliency-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/whitepapers/latest/introduction-aws-security/introduction-aws-security.pdf",
            "filename": "Introduction-to-AWS-Security.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── Security service FAQs ──────────────────
        {
            "url"     : "https://aws.amazon.com/iam/faqs/",
            "filename": "FAQ-IAM.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kms/faqs/",
            "filename": "FAQ-KMS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudtrail/faqs/",
            "filename": "FAQ-CloudTrail.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/guardduty/faqs/",
            "filename": "FAQ-GuardDuty.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/security-hub/faqs/",
            "filename": "FAQ-SecurityHub.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/inspector/faqs/",
            "filename": "FAQ-Inspector.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/macie/faqs/",
            "filename": "FAQ-Macie.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/waf/faqs/",
            "filename": "FAQ-WAF.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/shield/faqs/",
            "filename": "FAQ-Shield.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cognito/faqs/",
            "filename": "FAQ-Cognito.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/secrets-manager/faqs/",
            "filename": "FAQ-SecretsManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/certificate-manager/faqs/",
            "filename": "FAQ-CertificateManager.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/config/faqs/",
            "filename": "FAQ-Config.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/detective/faqs/",
            "filename": "FAQ-Detective.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/network-firewall/faqs/",
            "filename": "FAQ-NetworkFirewall.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # ANS-C01 — AWS Certified Advanced Networking Specialty
    # ══════════════════════════════════════════
    "ANS-C01": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-advnetworking-spec/AWS-Certified-Advanced-Networking-Specialty_Exam-Guide.pdf",
            "filename": "ANS-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-vpc-connectivity-options/aws-vpc-connectivity-options.pdf",
            "filename": "AWS-VPC-Connectivity-Options.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/building-scalable-secure-multi-vpc-network-infrastructure/building-scalable-secure-multi-vpc-network-infrastructure.pdf",
            "filename": "Building-Scalable-Secure-Multi-VPC.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-best-practices-ddos-resiliency/aws-best-practices-ddos-resiliency.pdf",
            "filename": "AWS-DDoS-Resiliency-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/hybrid-connectivity/hybrid-connectivity.pdf",
            "filename": "AWS-Hybrid-Connectivity.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-privatelink/aws-privatelink.pdf",
            "filename": "AWS-PrivateLink-Whitepaper.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── Networking service FAQs ────────────────
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/directconnect/faqs/",
            "filename": "FAQ-DirectConnect.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/transit-gateway/faqs/",
            "filename": "FAQ-TransitGateway.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/route53/faqs/",
            "filename": "FAQ-Route53.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/cloudfront/faqs/",
            "filename": "FAQ-CloudFront.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticloadbalancing/faqs/",
            "filename": "FAQ-ELB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/network-firewall/faqs/",
            "filename": "FAQ-NetworkFirewall.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/global-accelerator/faqs/",
            "filename": "FAQ-GlobalAccelerator.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/vpn/faqs/",
            "filename": "FAQ-VPN.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/api-gateway/faqs/",
            "filename": "FAQ-APIGateway.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # DAS-C01 — AWS Certified Data Analytics Specialty
    # ══════════════════════════════════════════
    "DAS-C01": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-data-analytics-specialty/AWS-Certified-Data-Analytics-Specialty_Exam-Guide.pdf",
            "filename": "DAS-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/big-data-analytics-options/big-data-analytics-options.pdf",
            "filename": "Big-Data-Analytics-Options.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/building-data-lakes/building-data-lakes.pdf",
            "filename": "Building-Data-Lakes-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-glue-best-practices-build-performant-data-pipeline/aws-glue-best-practices-build-performant-data-pipeline.pdf",
            "filename": "AWS-Glue-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── Data analytics service FAQs ────────────
        {
            "url"     : "https://aws.amazon.com/kinesis/faqs/",
            "filename": "FAQ-Kinesis.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/redshift/faqs/",
            "filename": "FAQ-Redshift.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/emr/faqs/",
            "filename": "FAQ-EMR.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/athena/faqs/",
            "filename": "FAQ-Athena.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/glue/faqs/",
            "filename": "FAQ-Glue.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/dynamodb/faqs/",
            "filename": "FAQ-DynamoDB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/opensearch-service/faqs/",
            "filename": "FAQ-OpenSearch.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/quicksight/faqs/",
            "filename": "FAQ-QuickSight.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/msk/faqs/",
            "filename": "FAQ-MSK.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lake-formation/faqs/",
            "filename": "FAQ-LakeFormation.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # MLS-C01 — AWS Certified Machine Learning Specialty
    # ══════════════════════════════════════════
    "MLS-C01": [
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/aws-certification/latest/machine-learning-specialty-01/machine-learning-specialty-01.pdf",
            "filename": "MLS-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/aws-caf-for-ai/aws-caf-for-ai.pdf",
            "filename": "AWS-CAF-for-AI.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/strategy-enterprise-ready-gen-ai-platform/strategy-enterprise-ready-gen-ai-platform.pdf",
            "filename": "Enterprise-Ready-GenAI-Platform.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/ml-best-practices-healthcare-life-sciences/ml-best-practices-healthcare-life-sciences.pdf",
            "filename": "ML-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf",
            "filename": "Well-Architected-Security-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/whitepapers/latest/building-data-lakes/building-data-lakes.pdf",
            "filename": "Building-Data-Lakes-on-AWS.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": False,
        },
        # ── ML Specialty service FAQs ──────────────
        {
            "url"     : "https://aws.amazon.com/sagemaker/faqs/",
            "filename": "FAQ-SageMaker.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/bedrock/faqs/",
            "filename": "FAQ-Bedrock.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rekognition/faqs/",
            "filename": "FAQ-Rekognition.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/comprehend/faqs/",
            "filename": "FAQ-Comprehend.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/forecast/faqs/",
            "filename": "FAQ-Forecast.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/personalize/faqs/",
            "filename": "FAQ-Personalize.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/translate/faqs/",
            "filename": "FAQ-Translate.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/transcribe/faqs/",
            "filename": "FAQ-Transcribe.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/polly/faqs/",
            "filename": "FAQ-Polly.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/lex/faqs/",
            "filename": "FAQ-Lex.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/textract/faqs/",
            "filename": "FAQ-Textract.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/glue/faqs/",
            "filename": "FAQ-Glue.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/kinesis/faqs/",
            "filename": "FAQ-Kinesis.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/emr/faqs/",
            "filename": "FAQ-EMR.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],

    # ══════════════════════════════════════════
    # PAS-C01 — AWS Certified SAP on AWS Specialty
    # ══════════════════════════════════════════
    "PAS-C01": [
        {
            "url"     : "https://d1.awsstatic.com/training-and-certification/docs-sap-on-aws-specialty/AWS-Certified-SAP-on-AWS-Specialty_Exam-Guide.pdf",
            "filename": "PAS-C01_Exam-Guide.pdf",
            "sha256"  : None,
            "category": "exam_guide",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/pdfs/sap/latest/general/general.pdf",
            "filename": "SAP-on-AWS-Best-Practices.pdf",
            "sha256"  : None,
            "category": "whitepaper",
            "required": True,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf",
            "filename": "AWS-Well-Architected-Framework.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        {
            "url"     : "https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/wellarchitected-reliability-pillar.pdf",
            "filename": "Well-Architected-Reliability-Pillar.pdf",
            "sha256"  : None,
            "category": "framework",
            "required": False,
        },
        # ── SAP on AWS service FAQs ────────────────
        {
            "url"     : "https://aws.amazon.com/ec2/faqs/",
            "filename": "FAQ-EC2.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/efs/faq/",
            "filename": "FAQ-EFS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/ebs/faqs/",
            "filename": "FAQ-EBS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/rds/faqs/",
            "filename": "FAQ-RDS.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/s3/faqs/",
            "filename": "FAQ-S3.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/vpc/faqs/",
            "filename": "FAQ-VPC.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/directconnect/faqs/",
            "filename": "FAQ-DirectConnect.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/elasticloadbalancing/faqs/",
            "filename": "FAQ-ELB.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/autoscaling/faqs/",
            "filename": "FAQ-AutoScaling.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
        {
            "url"     : "https://aws.amazon.com/backup/faqs/",
            "filename": "FAQ-Backup.html",
            "sha256"  : None,
            "category": "faq",
            "required": False,
        },
    ],
}


# ─────────────────────────────────────────────
# Download result dataclass
# ─────────────────────────────────────────────

@dataclass
class DownloadResult:
    """
    Outcome record for a single document download attempt.
    """
    cert_code:        str
    filename:         str
    url:              str
    category:         str
    required:         bool
    status:           str           = "pending"   # ok | skipped | failed
    file_size_bytes:  int           = 0
    duration_seconds: float         = 0.0
    checksum_ok:      Optional[bool]= None
    error_message:    str           = ""


@dataclass
class DownloadReport:
    """
    Aggregated results for a full certification download run.
    """
    cert_code:        str
    total:            int               = 0
    downloaded:       int               = 0
    skipped:          int               = 0
    failed:           int               = 0
    failed_required:  int               = 0
    total_bytes:      int               = 0
    duration_seconds: float             = 0.0
    results:          List[DownloadResult] = field(default_factory=list)


# ─────────────────────────────────────────────
# HTTP session factory
# ─────────────────────────────────────────────

def _build_session(
    max_retries:     int   = 3,
    backoff_factor:  float = 1.5,
    timeout:         int   = 60,
) -> requests.Session:
    """
    Build a requests.Session with automatic retry logic and
    a browser-like User-Agent header to avoid AWS CDN rejections.

    Retry policy:
      - 3 attempts on 500, 502, 503, 504 status codes.
      - Exponential backoff: 1.5s, 3s, 6s between retries.
      - Retries on connection errors and read timeouts.
    """
    session = requests.Session()

    retry_policy = Retry(
        total            = max_retries,
        backoff_factor   = backoff_factor,
        status_forcelist = [429, 500, 502, 503, 504],
        allowed_methods  = ["GET", "HEAD"],
        raise_on_status  = False,
    )

    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)

    session.headers.update({
        "User-Agent"     : (
            "Mozilla/5.0 (compatible; AWSExamDocDownloader/1.0; "
            "+https://github.com/aws-exam-gen)"
        ),
        "Accept"         : "application/pdf,text/html,*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
    })

    return session


# ─────────────────────────────────────────────
# Checksum verification
# ─────────────────────────────────────────────

def _sha256_file(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute the SHA-256 hex digest of a file without loading it
    fully into memory.  Uses 64 KB read chunks.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify_checksum(
    file_path:       Path,
    expected_sha256: Optional[str],
) -> Optional[bool]:
    """
    Verify a downloaded file's SHA-256 checksum.

    Returns:
      True  — checksum matches.
      False — checksum mismatch (file likely corrupt or updated).
      None  — no expected checksum provided (skip verification).
    """
    if expected_sha256 is None:
        return None

    actual = _sha256_file(file_path)
    match  = actual.lower() == expected_sha256.lower()

    if not match:
        log.warning(
            f"Checksum mismatch for '{file_path.name}':\n"
            f"  Expected: {expected_sha256}\n"
            f"  Actual  : {actual}"
        )
    else:
        log.debug(f"Checksum verified: '{file_path.name}' ✔")

    return match


# ─────────────────────────────────────────────
# File size formatting
# ─────────────────────────────────────────────

def _human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (KB / MB / GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{num_bytes / 1024 ** 3:.2f} GB"


# ─────────────────────────────────────────────
# Single file downloader
# ─────────────────────────────────────────────

def download_file(
    url:          str,
    dest_path:    Path,
    session:      requests.Session,
    progress:     Progress,
    task_id:      Any,
    force:        bool = False,
    dry_run:      bool = False,
    expected_sha256: Optional[str] = None,
) -> DownloadResult:
    """
    Download a single file from a URL to dest_path.

    Parameters
    ──────────
    url
        Direct PDF download URL.
    dest_path
        Full destination file path including filename.
    session
        Shared requests.Session with retry policy.
    progress
        Rich Progress instance for the download bar update.
    task_id
        Rich task ID to advance during streaming download.
    force
        If True, re-download even if the file already exists.
    dry_run
        If True, simulate the download without writing any files.
    expected_sha256
        If provided, verify the downloaded file's checksum.

    Returns
    ───────
    DownloadResult with status, size, duration, and checksum info.
    """
    filename = dest_path.name
    result   = DownloadResult(
        cert_code = dest_path.parent.name,
        filename  = filename,
        url       = url,
        category  = "",       # Filled by caller
        required  = False,    # Filled by caller
    )

    # ── Skip if already exists and not forced ──
    if dest_path.exists() and not force:
        existing_size = dest_path.stat().st_size
        log.debug(
            f"Skipping '{filename}' — already exists "
            f"({_human_size(existing_size)})."
        )
        result.status          = "skipped"
        result.file_size_bytes = existing_size
        progress.update(task_id, advance=existing_size)
        return result

    # ── Dry run ────────────────────────────────
    if dry_run:
        log.info(f"[DRY RUN] Would download: {url} → {dest_path}")
        result.status = "skipped"
        return result

    # ── Stream download ────────────────────────
    t_start = time.monotonic()

    try:
        response = session.get(url, stream=True, timeout=60)

        if response.status_code == 404:
            result.status        = "failed"
            result.error_message = (
                f"HTTP 404 — URL not found. "
                "The document may have been moved or renamed by AWS."
            )
            log.warning(f"404 Not Found: {url}")
            return result

        if response.status_code != 200:
            result.status        = "failed"
            result.error_message = (
                f"HTTP {response.status_code} — "
                f"{response.reason}"
            )
            log.warning(
                f"HTTP {response.status_code} for '{filename}': "
                f"{response.reason}"
            )
            return result

        # Get content length for progress bar if available
        content_length = int(
            response.headers.get("Content-Length", 0)
        )
        if content_length:
            progress.update(task_id, total=content_length)

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file first, then rename atomically
        # so interrupted downloads don't leave corrupt partial files
        tmp_path    = dest_path.with_suffix(".tmp")
        bytes_written = 0

        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    bytes_written += len(chunk)
                    progress.advance(task_id, advance=len(chunk))

        # Atomic rename
        tmp_path.rename(dest_path)

        result.file_size_bytes  = bytes_written
        result.duration_seconds = round(time.monotonic() - t_start, 2)

        # ── Checksum verification ──────────────
        result.checksum_ok = _verify_checksum(dest_path, expected_sha256)
        if result.checksum_ok is False:
            result.status        = "failed"
            result.error_message = (
                "SHA-256 checksum mismatch — file may be corrupt. "
                "Re-run with --force to re-download."
            )
            return result

        result.status = "ok"
        log.info(
            f"Downloaded '{filename}' "
            f"({_human_size(bytes_written)}) "
            f"in {result.duration_seconds:.1f}s."
        )

    except requests.exceptions.ConnectionError as conn_err:
        result.status        = "failed"
        result.error_message = f"Connection error: {conn_err}"
        log.error(f"Connection error downloading '{filename}': {conn_err}")

    except requests.exceptions.Timeout:
        result.status        = "failed"
        result.error_message = "Request timed out after 60 seconds."
        log.error(f"Timeout downloading '{filename}'.")

    except requests.exceptions.RequestException as req_err:
        result.status        = "failed"
        result.error_message = str(req_err)
        log.error(f"Request error downloading '{filename}': {req_err}")

    except OSError as io_err:
        result.status        = "failed"
        result.error_message = f"File system error: {io_err}"
        log.error(f"IO error writing '{filename}': {io_err}")
        # Clean up partial temp file
        tmp_path = dest_path.with_suffix(".tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return result


# ─────────────────────────────────────────────
# Certification download orchestrator
# ─────────────────────────────────────────────

def download_cert_documents(
    cert_code:    str,
    output_dir:   Path,
    force:        bool = False,
    dry_run:      bool = False,
    required_only:bool = False,
) -> DownloadReport:
    """
    Download all documents for a single certification code.

    Parameters
    ──────────
    cert_code
        AWS certification code (e.g. 'SAA-C03').
    output_dir
        Root data directory. Files are saved to output_dir/cert_code/.
    force
        Re-download files that already exist locally.
    dry_run
        Log what would be downloaded without writing any files.
    required_only
        Only download documents marked required=True.

    Returns
    ───────
    DownloadReport with per-file results and aggregate statistics.
    """
    cert_code = cert_code.upper().strip()

    if cert_code not in DOCUMENT_CATALOGUE:
        raise ValueError(
            f"No document catalogue entry for '{cert_code}'. "
            f"Available: {', '.join(sorted(DOCUMENT_CATALOGUE.keys()))}"
        )

    documents   = DOCUMENT_CATALOGUE[cert_code]
    dest_dir    = output_dir / cert_code
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Apply required_only filter
    if required_only:
        documents = [d for d in documents if d["required"]]
        log.info(
            f"required_only=True: downloading "
            f"{len(documents)} required document(s) for {cert_code}."
        )

    report = DownloadReport(
        cert_code = cert_code,
        total     = len(documents),
    )

    session    = _build_session()
    run_start  = time.monotonic()

    console.print(
        Panel(
            f"[bold]Certification:[/bold] {cert_code}\n"
            f"[bold]Destination  :[/bold] {dest_dir}\n"
            f"[bold]Documents    :[/bold] {len(documents)}\n"
            f"[bold]Force        :[/bold] {force}\n"
            f"[bold]Dry run      :[/bold] {dry_run}",
            title        = f"[bold blue]Downloading {cert_code} Documents[/bold blue]",
            border_style = "blue",
            expand       = False,
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console  = console,
        transient= False,
    ) as progress:

        for doc in documents:
            filename  = doc["filename"]
            url       = doc["url"]
            category  = doc["category"]
            required  = doc["required"]
            sha256    = doc.get("sha256")
            dest_path = dest_dir / filename

            # Add a Rich task for this file
            task_id = progress.add_task(
                description = f"[cyan]{filename}",
                total       = None,   # Unknown until Content-Length header
            )

            result          = download_file(
                url             = url,
                dest_path       = dest_path,
                session         = session,
                progress        = progress,
                task_id         = task_id,
                force           = force,
                dry_run         = dry_run,
                expected_sha256 = sha256,
            )

            # Fill in fields the downloader doesn't know about
            result.category = category
            result.required = required
            result.cert_code= cert_code

            report.results.append(result)

            # ── Update aggregate counters ──────
            if result.status == "ok":
                report.downloaded    += 1
                report.total_bytes   += result.file_size_bytes
            elif result.status == "skipped":
                report.skipped       += 1
                report.total_bytes   += result.file_size_bytes
            elif result.status == "failed":
                report.failed        += 1
                if required:
                    report.failed_required += 1
                log.error(
                    f"FAILED ({'required' if required else 'optional'}): "
                    f"'{filename}' — {result.error_message}"
                )

            # Mark task complete regardless of outcome
            progress.update(task_id, completed=True)

    report.duration_seconds = round(time.monotonic() - run_start, 2)
    return report


# ─────────────────────────────────────────────
# Report display
# ─────────────────────────────────────────────

def _print_download_report(report: DownloadReport) -> None:
    """
    Render a Rich table and summary panel for a DownloadReport.
    """
    # ── Per-file results table ─────────────────
    table = Table(
        title        = f"Download Results — {report.cert_code}",
        show_header  = True,
        header_style = "bold magenta",
        show_lines   = True,
    )

    table.add_column("Filename",   style="cyan",   width=48)
    table.add_column("Category",   style="blue",   width=12)
    table.add_column("Required",   style="white",  width=10, justify="center")
    table.add_column("Size",       style="green",  width=10, justify="right")
    table.add_column("Duration",   style="yellow", width=10, justify="right")
    table.add_column("Checksum",   style="white",  width=10, justify="center")
    table.add_column("Status",     style="white",  width=10, justify="center")

    for result in report.results:
        status_str = {
            "ok"     : "[green]✔ OK[/green]",
            "skipped": "[yellow]⏭ Skip[/yellow]",
            "failed" : "[red]✘ FAIL[/red]",
        }.get(result.status, result.status)

        checksum_str = {
            True : "[green]✔[/green]",
            False: "[red]✘[/red]",
            None : "[dim]N/A[/dim]",
        }.get(result.checksum_ok, "[dim]N/A[/dim]")

        req_str = (
            "[white]Yes[/white]" if result.required
            else "[dim]No[/dim]"
        )

        size_str = (
            _human_size(result.file_size_bytes)
            if result.file_size_bytes > 0
            else "—"
        )

        dur_str = (
            f"{result.duration_seconds:.1f}s"
            if result.duration_seconds > 0
            else "—"
        )

        table.add_row(
            result.filename[:46],
            result.category,
            req_str,
            size_str,
            dur_str,
            checksum_str,
            status_str,
        )

    console.print(table)

    # ── Summary panel ──────────────────────────
    failed_names = [
        r.filename for r in report.results if r.status == "failed"
    ]
    failed_str = (
        "\n  ".join(failed_names) if failed_names else "None"
    )

    summary = (
        f"[bold]Certification   :[/bold]  {report.cert_code}\n"
        f"[bold]Total documents :[/bold]  {report.total}\n"
        f"[bold]Downloaded      :[/bold]  [green]{report.downloaded}[/green]\n"
        f"[bold]Skipped         :[/bold]  [yellow]{report.skipped}[/yellow]"
        f"  (already existed)\n"
        f"[bold]Failed          :[/bold]  [red]{report.failed}[/red]"
        + (
            f"  ([red]{report.failed_required} required[/red])"
            if report.failed_required else ""
        ) + "\n"
        f"[bold]Total size      :[/bold]  {_human_size(report.total_bytes)}\n"
        f"[bold]Duration        :[/bold]  {report.duration_seconds:.1f}s\n"
        f"[bold]Failed files    :[/bold]\n  {failed_str}"
    )

    border = "red" if report.failed_required > 0 else "blue"
    title  = (
        "[red]✘ Download Complete (with required failures)[/red]"
        if report.failed_required > 0
        else "[bold blue]✔ Download Complete[/bold blue]"
    )

    console.print(
        Panel(
            summary,
            title        = title,
            border_style = border,
            expand       = False,
        )
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@click.command()
@click.option(
    "--cert",
    required = True,
    metavar  = "CERT_CODE",
    help     = (
        "Certification code to download documents for "
        "(e.g. SAA-C03), or 'all' to download every certification."
    ),
)
@click.option(
    "--output-dir",
    "output_dir",
    default  = None,
    metavar  = "DIR",
    help     = (
        "Root directory for downloaded PDFs. "
        "Files are saved to <output-dir>/<CERT_CODE>/. "
        "Defaults to ./data/."
    ),
)
@click.option(
    "--force",
    is_flag  = True,
    default  = False,
    help     = "Re-download files that already exist locally.",
)
@click.option(
    "--dry-run",
    is_flag  = True,
    default  = False,
    help     = "Print what would be downloaded without writing any files.",
)
@click.option(
    "--required-only",
    is_flag  = True,
    default  = False,
    help     = "Only download documents marked as required (skip optional).",
)
@click.option(
    "--list",
    "list_certs",
    is_flag  = True,
    default  = False,
    help     = "List all certifications with available documents and exit.",
)
@click.option(
    "--log-level",
    default  = "INFO",
    type     = click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        case_sensitive=False,
    ),
    show_default=True,
    help     = "Logging verbosity.",
)
def main(
    cert:          str,
    output_dir:    Optional[str],
    force:         bool,
    dry_run:       bool,
    required_only: bool,
    list_certs:    bool,
    log_level:     str,
) -> None:
    """
    \b
    ╔══════════════════════════════════════════════════════╗
    ║   AWS Exam Document Downloader                       ║
    ║   Downloads official PDFs for ChromaDB ingestion     ║
    ╚══════════════════════════════════════════════════════╝

    \b
    Downloads official AWS exam guides, whitepapers,
    Well-Architected pillars, and FAQs into the correct
    directory structure for use with the ingest pipeline.

    \b
    Examples:
      python download_docs.py --cert SAA-C03
      python download_docs.py --cert all
      python download_docs.py --cert SAA-C03 --dry-run
      python download_docs.py --cert SAA-C03 --force
      python download_docs.py --cert SAA-C03 --required-only
      python download_docs.py --list
    """
    setup_logging(log_level)

    # ── Resolve output directory ───────────────
    root_dir = Path(output_dir).resolve() if output_dir else DATA_DIR
    root_dir.mkdir(parents=True, exist_ok=True)

    # ── List mode ──────────────────────────────
    if list_certs:
        table = Table(
            title        = "Available Document Catalogues",
            show_header  = True,
            header_style = "bold magenta",
            show_lines   = True,
        )
        table.add_column("Cert Code",    style="bold cyan",  width=12)
        table.add_column("Total Docs",   style="white",      width=12, justify="right")
        table.add_column("Required",     style="green",      width=10, justify="right")
        table.add_column("Optional",     style="yellow",     width=10, justify="right")
        table.add_column("Categories",   style="blue",       width=40)

        for code, docs in sorted(DOCUMENT_CATALOGUE.items()):
            required_count = sum(1 for d in docs if d["required"])
            optional_count = len(docs) - required_count
            categories     = sorted(set(d["category"] for d in docs))
            table.add_row(
                code,
                str(len(docs)),
                str(required_count),
                str(optional_count),
                ", ".join(categories),
            )

        console.print(table)
        console.print(
            f"\n[dim]Total certifications with catalogues: "
            f"{len(DOCUMENT_CATALOGUE)}[/dim]\n\n"
            "[dim]To download a certification's documents:[/dim]\n"
            "  [bold]python download_docs.py --cert <CODE>[/bold]\n\n"
            "[dim]To download all certifications:[/dim]\n"
            "  [bold]python download_docs.py --cert all[/bold]"
        )
        sys.exit(0)

    # ── Resolve cert codes to process ─────────
    if cert.lower() == "all":
        cert_codes = sorted(DOCUMENT_CATALOGUE.keys())
        console.print(
            f"[bold]Downloading documents for ALL "
            f"{len(cert_codes)} certification(s)…[/bold]\n"
        )
    else:
        cert_code = cert.upper().strip()
        if cert_code not in DOCUMENT_CATALOGUE:
            console.print(
                Panel(
                    f"[bold red]No document catalogue found for "
                    f"'{cert_code}'.[/bold red]\n\n"
                    f"Available certifications:\n"
                    f"  {', '.join(sorted(DOCUMENT_CATALOGUE.keys()))}\n\n"
                    f"Run [bold]python download_docs.py --list[/bold] "
                    f"for details.",
                    title        = "[red]Unknown Certification[/red]",
                    border_style = "red",
                    expand       = False,
                )
            )
            sys.exit(1)
        cert_codes = [cert_code]

    # ── Dry run banner ─────────────────────────
    if dry_run:
        console.print(
            Panel(
                "[bold yellow]DRY RUN MODE — no files will be written.\n"
                "All download actions will be logged but not executed.[/bold yellow]",
                title        = "[yellow]Dry Run[/yellow]",
                border_style = "yellow",
                expand       = False,
            )
        )

    # ── Download loop ──────────────────────────
    all_reports: List[DownloadReport] = []
    overall_start = time.monotonic()

    for cert_code in cert_codes:
        try:
            report = download_cert_documents(
                cert_code     = cert_code,
                output_dir    = root_dir,
                force         = force,
                dry_run       = dry_run,
                required_only = required_only,
            )
            all_reports.append(report)
            _print_download_report(report)

        except ValueError as val_err:
            console.print(
                f"[red]✘ Skipping '{cert_code}': {val_err}[/red]"
            )
        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Download interrupted by user.[/yellow]"
            )
            sys.exit(130)
        except Exception as exc:
            log.debug(exc, exc_info=True)
            console.print(
                f"[red]✘ Unexpected error downloading '{cert_code}': "
                f"{exc}[/red]"
            )

    # ── Multi-cert aggregate summary ───────────
    if len(all_reports) > 1:
        _print_aggregate_summary(all_reports, time.monotonic() - overall_start)

    # ── Next steps hint ────────────────────────
    if not dry_run:
        successful_certs = [
            r.cert_code for r in all_reports
            if r.failed_required == 0 and r.downloaded + r.skipped > 0
        ]
        if successful_certs:
            codes_str = " ".join(f"--cert {c}" for c in successful_certs)
            console.print(
                Panel(
                    "[bold green]Documents ready for ingestion.[/bold green]\n\n"
                    "Run the ingest pipeline to embed these documents:\n\n"
                    + "\n".join(
                        f"  [bold]python main.py ingest --cert {c}[/bold]"
                        for c in successful_certs
                    ),
                    title        = "[green]Next Steps[/green]",
                    border_style = "green",
                    expand       = False,
                )
            )

    # ── Exit code ──────────────────────────────
    # Non-zero exit if any required document failed across any cert
    total_required_failures = sum(
        r.failed_required for r in all_reports
    )
    sys.exit(1 if total_required_failures > 0 else 0)


# ─────────────────────────────────────────────
# Aggregate summary (multi-cert runs)
# ─────────────────────────────────────────────

def _print_aggregate_summary(
    reports:          List[DownloadReport],
    total_duration:   float,
) -> None:
    """
    Print a consolidated summary table across all certifications
    when --cert all is used.
    """
    table = Table(
        title        = "Aggregate Download Summary — All Certifications",
        show_header  = True,
        header_style = "bold magenta",
        show_lines   = True,
    )

    table.add_column("Cert Code",   style="bold cyan", width=12)
    table.add_column("Total",       style="white",     width=8,  justify="right")
    table.add_column("Downloaded",  style="green",     width=12, justify="right")
    table.add_column("Skipped",     style="yellow",    width=10, justify="right")
    table.add_column("Failed",      style="red",       width=8,  justify="right")
    table.add_column("Req. Failed", style="red",       width=12, justify="right")
    table.add_column("Total Size",  style="blue",      width=12, justify="right")
    table.add_column("Duration",    style="white",     width=10, justify="right")
    table.add_column("Status",      style="white",     width=10, justify="center")

    grand_total       = 0
    grand_downloaded  = 0
    grand_skipped     = 0
    grand_failed      = 0
    grand_req_failed  = 0
    grand_bytes       = 0

    for report in reports:
        status_str = (
            "[green]✔ OK[/green]"
            if report.failed_required == 0
            else "[red]✘ FAIL[/red]"
        )
        table.add_row(
            report.cert_code,
            str(report.total),
            str(report.downloaded),
            str(report.skipped),
            str(report.failed),
            str(report.failed_required),
            _human_size(report.total_bytes),
            f"{report.duration_seconds:.1f}s",
            status_str,
        )
        grand_total      += report.total
        grand_downloaded += report.downloaded
        grand_skipped    += report.skipped
        grand_failed     += report.failed
        grand_req_failed += report.failed_required
        grand_bytes      += report.total_bytes

    # Totals row
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{grand_total}[/bold]",
        f"[bold green]{grand_downloaded}[/bold green]",
        f"[bold yellow]{grand_skipped}[/bold yellow]",
        f"[bold red]{grand_failed}[/bold red]",
        f"[bold red]{grand_req_failed}[/bold red]",
        f"[bold]{_human_size(grand_bytes)}[/bold]",
        f"[bold]{total_duration:.1f}s[/bold]",
        (
            "[bold green]✔ OK[/bold green]"
            if grand_req_failed == 0
            else "[bold red]✘ FAIL[/bold red]"
        ),
    )

    console.print(table)

    console.print(
        Panel(
            f"[bold]Certifications processed:[/bold]  {len(reports)}\n"
            f"[bold]Total documents        :[/bold]  {grand_total}\n"
            f"[bold]Successfully downloaded:[/bold]  "
            f"[green]{grand_downloaded}[/green]\n"
            f"[bold]Already existed (skip) :[/bold]  "
            f"[yellow]{grand_skipped}[/yellow]\n"
            f"[bold]Failed downloads       :[/bold]  "
            f"[red]{grand_failed}[/red]\n"
            f"[bold]Failed required docs   :[/bold]  "
            f"[red]{grand_req_failed}[/red]\n"
            f"[bold]Total data transferred :[/bold]  "
            f"{_human_size(grand_bytes)}\n"
            f"[bold]Total duration         :[/bold]  "
            f"{total_duration:.1f}s",
            title        = "[bold blue]All Certifications — Final Summary[/bold blue]",
            border_style = "blue" if grand_req_failed == 0 else "red",
            expand       = False,
        )
    )


# ─────────────────────────────────────────────
# Catalogue utilities (importable helpers)
# ─────────────────────────────────────────────

def get_catalogue_for_cert(cert_code: str) -> List[Dict]:
    """
    Return the document catalogue list for a given certification code.

    Parameters
    ──────────
    cert_code
        AWS certification code (e.g. 'SAA-C03').

    Returns
    ───────
    List of document definition dicts.

    Raises ValueError if cert_code is not in the catalogue.
    """
    cert_code = cert_code.upper().strip()
    if cert_code not in DOCUMENT_CATALOGUE:
        raise ValueError(
            f"No catalogue entry for '{cert_code}'. "
            f"Available: {', '.join(sorted(DOCUMENT_CATALOGUE.keys()))}"
        )
    return DOCUMENT_CATALOGUE[cert_code]


def list_supported_certs() -> List[str]:
    """
    Return a sorted list of all certification codes that have
    a document catalogue defined.
    """
    return sorted(DOCUMENT_CATALOGUE.keys())


def get_required_documents(cert_code: str) -> List[Dict]:
    """
    Return only the required documents for a certification.
    Useful for programmatic checks before running the ingest pipeline.
    """
    return [
        doc for doc in get_catalogue_for_cert(cert_code)
        if doc["required"]
    ]


def check_local_documents(
    cert_code:  str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Check which catalogue documents are already present on disk
    for a given certification without downloading anything.

    Parameters
    ──────────
    cert_code
        AWS certification code.
    output_dir
        Root data directory. Defaults to DATA_DIR from config.

    Returns
    ───────
    Dict with keys:
      cert_code       : str
      dest_dir        : str
      total           : int
      present         : List[str]   — filenames found on disk
      missing         : List[str]   — filenames not found
      missing_required: List[str]   — required filenames not found
      ready_to_ingest : bool        — True if all required docs present
    """
    root  = output_dir or DATA_DIR
    docs  = get_catalogue_for_cert(cert_code)
    dest  = root / cert_code.upper()

    present:          List[str] = []
    missing:          List[str] = []
    missing_required: List[str] = []

    for doc in docs:
        path = dest / doc["filename"]
        if path.exists() and path.stat().st_size > 0:
            present.append(doc["filename"])
        else:
            missing.append(doc["filename"])
            if doc["required"]:
                missing_required.append(doc["filename"])

    return {
        "cert_code"       : cert_code.upper(),
        "dest_dir"        : str(dest),
        "total"           : len(docs),
        "present"         : present,
        "missing"         : missing,
        "missing_required": missing_required,
        "ready_to_ingest" : len(missing_required) == 0,
    }


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
