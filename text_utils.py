"""
text_utils.py
──────────────
Fixes "shouted" ALL CAPS text that local LLMs occasionally produce when
echoing option text inside explanations (e.g. "A. USE AWS LAMBDA TO ...").

The only public entry point is `fix_shouting_caps`, which finds runs of
three or more consecutive all-caps words and rewrites them to normal
sentence case, while restoring the correct mixed case for known AWS
acronyms and product names. Text outside those runs (including short,
intentional emphasis like "Select TWO") is left untouched.
"""

from __future__ import annotations

import re
from typing import Dict, List

# ─────────────────────────────────────────────
# Known casing exceptions
# ─────────────────────────────────────────────

# Acronyms/initialisms that must stay fully uppercase.
ACRONYMS = {
    "AWS", "EC2", "S3", "VPC", "IAM", "KMS", "RDS", "ARN",
    "API", "CLI", "JSON", "XML", "YAML", "SDK", "VPN", "DNS",
    "CIDR", "TCP", "UDP", "IP", "HTTP", "HTTPS", "SSL", "TLS", "CPU",
    "GPU", "ML", "AI", "ETL", "ELB", "ALB", "NLB", "ECS", "EKS", "ECR",
    "SNS", "SQS", "WAF", "ACM", "STS", "IOT", "CDN", "ASG", "EBS", "EFS",
    "MFA", "RBAC", "ABAC", "SLA", "RPO", "RTO", "KPI", "PDF", "URL", "ID",
    "IDS", "CSPM", "SFTP", "HSM", "ENI", "NAT", "IGW", "VGW", "TGW",
    "ACL", "IOPS", "GB", "TB", "MB", "KB", "PB", "CI", "CD",
    "EMR", "KDS", "DMS", "DR", "HA", "SSO", "SAML", "OIDC", "JWT",
    "SOAP", "GDPR", "HIPAA", "PCI", "SOC", "ISO", "FIPS", "SES", "SIEM",
    "XSS", "CSRF", "EIP", "IPV4", "IPV6", "MTU", "BGP", "ASN", "MSK",
    "QLDB", "FSX", "ADFS", "LDAP", "SCP", "OU", "AZ",
    "RAM", "SSM", "WAFV2", "NACL", "VIF", "DX",
    "CMK", "IV", "CAF", "SSE", "X", "SAM", "DSS",
}

# Single-word AWS product names with distinctive internal capitalisation.
CANONICAL_WORDS: Dict[str, str] = {
    "amazon":         "Amazon",
    "sagemaker":      "SageMaker",
    "dynamodb":       "DynamoDB",
    "cloudformation": "CloudFormation",
    "cloudtrail":     "CloudTrail",
    "cloudwatch":     "CloudWatch",
    "cloudfront":     "CloudFront",
    "cloudhsm":       "CloudHSM",
    "codepipeline":   "CodePipeline",
    "codebuild":      "CodeBuild",
    "codedeploy":     "CodeDeploy",
    "codecommit":     "CodeCommit",
    "codestar":       "CodeStar",
    "codeartifact":   "CodeArtifact",
    "codeguru":       "CodeGuru",
    "codewhisperer":  "CodeWhisperer",
    "guardduty":      "GuardDuty",
    "quicksight":     "QuickSight",
    "redshift":       "Redshift",
    "athena":         "Athena",
    "glue":           "Glue",
    "lambda":         "Lambda",
    "fargate":        "Fargate",
    "aurora":         "Aurora",
    "snowball":       "Snowball",
    "snowmobile":     "Snowmobile",
    "outposts":       "Outposts",
    "lightsail":      "Lightsail",
    "elasticache":    "ElastiCache",
    "appsync":        "AppSync",
    "amplify":        "Amplify",
    "eventbridge":    "EventBridge",
    "opensearch":     "OpenSearch",
    "macie":          "Macie",
    "inspector":      "Inspector",
    "detective":      "Detective",
    "shield":         "Shield",
    "organizations":  "Organizations",
    "backup":         "Backup",
    "datasync":       "DataSync",
    "kinesis":        "Kinesis",
    "batch":          "Batch",
    "cognito":        "Cognito",
    "wavelength":     "Wavelength",
    "chime":          "Chime",
    "connect":        "Connect",
    "pinpoint":       "Pinpoint",
    "appflow":        "AppFlow",
    "datazone":       "DataZone",
    "clarify":        "Clarify",
    "bedrock":        "Bedrock",
    "comprehend":     "Comprehend",
    "rekognition":    "Rekognition",
    "textract":       "Textract",
    "transcribe":     "Transcribe",
    "translate":      "Translate",
    "polly":          "Polly",
    "lex":            "Lex",
    "forecast":       "Forecast",
    "personalize":    "Personalize",
    "artifact":       "Artifact",
    "config":         "Config",
    "ray":            "Ray",
    "devops":         "DevOps",
    "well":           "Well",
    "architected":    "Architected",
    "glacier":        "Glacier",
    "privatelink":    "PrivateLink",
}

# Multi-word AWS product names, keyed by their upper-cased, single-spaced form.
CANONICAL_PHRASES: Dict[str, str] = {
    "ELASTIC BEANSTALK":       "Elastic Beanstalk",
    "TRANSIT GATEWAY":         "Transit Gateway",
    "DIRECT CONNECT":          "Direct Connect",
    "SECURITY HUB":            "Security Hub",
    "SECRETS MANAGER":         "Secrets Manager",
    "STORAGE GATEWAY":         "Storage Gateway",
    "TRANSFER FAMILY":         "Transfer Family",
    "KEY MANAGEMENT SERVICE":  "Key Management Service",
    "CERTIFICATE MANAGER":     "Certificate Manager",
    "SYSTEMS MANAGER":         "Systems Manager",
    "CONTROL TOWER":           "Control Tower",
    "SERVICE CATALOG":         "Service Catalog",
    "GLOBAL ACCELERATOR":      "Global Accelerator",
    "FIREWALL MANAGER":        "Firewall Manager",
    "PARAMETER STORE":         "Parameter Store",
    "STEP FUNCTIONS":          "Step Functions",
    "LAKE FORMATION":          "Lake Formation",
    "APP RUNNER":              "App Runner",
    "CLOUD MAP":               "Cloud Map",
    "NAT GATEWAY":             "NAT Gateway",
    "INTERNET GATEWAY":        "Internet Gateway",
    "VIRTUAL PRIVATE GATEWAY": "Virtual Private Gateway",
    "API GATEWAY":             "API Gateway",
    "AUTO SCALING":            "Auto Scaling",
    "SNOWBALL EDGE":           "Snowball Edge",
    "DIRECTORY SERVICE":       "Directory Service",
    "ELEMENTAL MEDIACONVERT":  "Elemental MediaConvert",
    "ELEMENTAL MEDIA CONVERT": "Elemental MediaConvert",
    "ELEMENTAL MEDIALIVE":     "Elemental MediaLive",
    "LOCAL ZONES":             "Local Zones",
    "SITE-TO-SITE VPN":        "Site-to-Site VPN",
    "CLIENT VPN":              "Client VPN",
    "DATA FIREHOSE":           "Data Firehose",
    "KINESIS DATA FIREHOSE":   "Kinesis Data Firehose",
    "KINESIS DATA STREAMS":    "Kinesis Data Streams",
    "AMAZON MACIE":            "Amazon Macie",
    "NETWORK FIREWALL":        "Network Firewall",
    "DEVOPS GURU":             "DevOps Guru",
    "DATABASE MIGRATION SERVICE": "Database Migration Service",
    "IDENTITY AND ACCESS MANAGEMENT": "Identity and Access Management",
    "SAGE MAKER":              "SageMaker",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")
_LABEL_RE = re.compile(r"^([A-E])[.)]\s+")

_PLACEHOLDER = "{}"
_PLACEHOLDER_RE = re.compile("(\\d+)")


def _phrase_pattern() -> "re.Pattern[str]":
    # Longest phrases first so multi-word matches win over shorter overlaps.
    phrases = sorted(CANONICAL_PHRASES, key=len, reverse=True)
    body = "|".join(
        r"\s+".join(re.escape(word) for word in phrase.split(" "))
        for phrase in phrases
    )
    return re.compile(rf"\b(?:{body})\b")


_PHRASE_PATTERN = _phrase_pattern()


def _restore_word(word: str) -> str:
    lower = word.lower()
    upper = word.upper()
    if lower in CANONICAL_WORDS:
        return CANONICAL_WORDS[lower]
    if upper in ACRONYMS:
        return upper
    if upper.endswith("S") and len(upper) > 2 and upper[:-1] in ACRONYMS:
        return upper[:-1] + "s"
    return lower


def _is_safe_token(word: str) -> bool:
    """True if restoring this token's case is unambiguous (no generic
    English word is being re-cased, only a known acronym/product name)."""
    lower = word.lower()
    upper = word.upper()
    if lower in CANONICAL_WORDS or upper in ACRONYMS:
        return True
    return upper.endswith("S") and len(upper) > 2 and upper[:-1] in ACRONYMS


def _sentence_case(text: str, force_first: bool) -> str:
    chars = list(text)
    capitalize_next = force_first
    for i, ch in enumerate(chars):
        if capitalize_next and ch.isalpha():
            chars[i] = ch.upper()
            capitalize_next = False
        elif ch in ".!?":
            capitalize_next = True
        elif ch == "\n":
            capitalize_next = True
    return "".join(chars)


def _fix_segment(segment: str, force_first: bool) -> str:
    """Rewrite one maximal all-caps run to normal mixed case."""
    placeholders: List[str] = []

    def _stash_phrase(m: "re.Match[str]") -> str:
        placeholders.append(CANONICAL_PHRASES[re.sub(r"\s+", " ", m.group(0))])
        return _PLACEHOLDER.format(len(placeholders) - 1)

    text = _PHRASE_PATTERN.sub(_stash_phrase, segment) if CANONICAL_PHRASES else segment
    text = _WORD_RE.sub(lambda m: _restore_word(m.group(0)), text)
    text = _PLACEHOLDER_RE.sub(lambda m: placeholders[int(m.group(1))], text)
    text = _sentence_case(text, force_first)
    return _restore_option_letter_lists(text)


def _restore_option_letter_lists(text: str) -> str:
    """
    Re-uppercase single letters a-e when they're clearly referencing
    answer options in a list, e.g. "options a, b, and d" -> "... A, B, and D".
    Bare articles like "a transit gateway" are left alone.
    """
    text = re.sub(r"\b[a-e]\b(?=\s*,)", lambda m: m.group(0).upper(), text)
    text = re.sub(r"(?<=,\s)[a-e]\b", lambda m: m.group(0).upper(), text)
    text = re.sub(
        r"\b(and|or)\s+([a-e])\b",
        lambda m: f"{m.group(1)} {m.group(2).upper()}",
        text,
    )
    return text


def _is_cap_token(token: str) -> bool:
    return token == token.upper()


def _run_qualifies(run: List["re.Match[str]"]) -> bool:
    """
    A run of all-caps tokens needs fixing if it's 3+ words long (almost
    certainly shouted prose), or - regardless of length - every token is
    a known acronym/product name, in which case restoring its canonical
    case carries no risk of mangling ordinary words.
    """
    if not run:
        return False
    if len(run) >= 3:
        return True
    return all(_is_safe_token(tok.group(0)) for tok in run)


def _fix_text_runs(text: str, start_is_boundary: bool) -> str:
    """
    Find maximal runs of 3+ consecutive all-caps words in `text` and
    rewrite each one via `_fix_segment`. Everything else is untouched.
    """
    tokens = list(_WORD_RE.finditer(text))
    if not tokens:
        return text

    runs: List[List["re.Match[str]"]] = []
    current: List["re.Match[str]"] = []
    prev_end = 0

    for tok in tokens:
        gap = text[prev_end:tok.start()]
        cap = _is_cap_token(tok.group(0))
        # A run breaks if this token isn't all-caps, or a lowercase letter
        # appears in the gap since the previous token (i.e. we've drifted
        # into normal prose).
        if cap and not re.search(r"[a-z]", gap):
            current.append(tok)
        else:
            if _run_qualifies(current):
                runs.append(current)
            current = [tok] if cap else []
        prev_end = tok.end()
    if _run_qualifies(current):
        runs.append(current)

    if not runs:
        return text

    out = []
    cursor = 0
    for run in runs:
        run_start = run[0].start()
        run_end = run[-1].end()
        out.append(text[cursor:run_start])

        preceding = text[:run_start].rstrip()
        force_first = (
            (run_start == 0 and start_is_boundary)
            or preceding == ""
            or preceding[-1] in ".!?"
            or "\n" in text[max(0, run_start - 2):run_start]
        )
        out.append(_fix_segment(text[run_start:run_end], force_first))
        cursor = run_end
    out.append(text[cursor:])
    return "".join(out)


def _fix_paragraph(paragraph: str) -> str:
    label_match = _LABEL_RE.match(paragraph)
    if label_match:
        prefix = label_match.group(0)
        rest = paragraph[label_match.end():]
        return prefix + _fix_text_runs(rest, start_is_boundary=True)
    return _fix_text_runs(paragraph, start_is_boundary=True)


def fix_shouting_caps(text: str) -> str:
    """
    Rewrite ALL CAPS runs of 3+ words in `text` to normal sentence case,
    preserving correct casing for known AWS acronyms and product names.
    Short all-caps phrases (e.g. "Select TWO") are left untouched.
    """
    if not text or not isinstance(text, str):
        return text
    paragraphs = text.split("\n\n")
    return "\n\n".join(_fix_paragraph(p) for p in paragraphs)


def has_shouting_caps(text: str) -> bool:
    """True if `text` contains a run of 3+ consecutive all-caps words."""
    if not text or not isinstance(text, str):
        return False
    return fix_shouting_caps(text) != text
