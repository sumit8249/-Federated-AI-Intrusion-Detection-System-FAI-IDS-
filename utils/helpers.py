"""
Utility Helpers and Threat Intelligence Mappings for FAI-IDS
Spec Reference: Section 6 (Sample Attack Types) & Section 12 (Workflow)
"""

from typing import Dict, Any

# Threat severity and mitigation mapping for all 15 attack classes in CICIDS2017
THREAT_INTELLIGENCE: Dict[str, Dict[str, Any]] = {
    "BENIGN": {
        "severity": "Info",
        "color": "#10b981", # emerald
        "description": "Legitimate, benign network traffic exhibiting standard communication patterns.",
        "mitigation": "No action required. Flow marked as safe."
    },
    "DoS Hulk": {
        "severity": "Critical",
        "color": "#ef4444", # red
        "description": "High-volume HTTP request flood designed to exhaust web server threads and memory.",
        "mitigation": "Deploy rate-limiting at reverse proxy (Nginx/Cloudflare) and enable IP throttling."
    },
    "DDoS": {
        "severity": "Critical",
        "color": "#dc2626", # dark red
        "description": "Distributed Denial of Service orchestrated across multiple bot hosts to saturate network bandwidth.",
        "mitigation": "Activate upstream DDoS scrubbing service and block traffic at Border Gateway Protocol (BGP)."
    },
    "PortScan": {
        "severity": "Medium",
        "color": "#f59e0b", # amber
        "description": "Host/port enumeration reconnaissance probing open TCP/UDP services for vulnerabilities.",
        "mitigation": "Enforce firewall drop rules, close unused ports, and enable fail2ban automatic ban."
    },
    "DoS GoldenEye": {
        "severity": "High",
        "color": "#f97316", # orange
        "description": "Layer 7 DoS attack utilizing keep-alive and cache-control headers to exhaust connection pools.",
        "mitigation": "Configure connection timeout limits and enforce aggressive keep-alive termination."
    },
    "FTP-Patator": {
        "severity": "High",
        "color": "#e11d48", # rose
        "description": "Automated brute-force password guessing attack targeting FTP port 21.",
        "mitigation": "Enforce account lockouts after 3 attempts, enable TLS/SFTP, and disable anonymous FTP."
    },
    "SSH-Patator": {
        "severity": "High",
        "color": "#e11d48", # rose
        "description": "Dictionary attack attempting unauthorized SSH logins on port 22.",
        "mitigation": "Disable root login, disable password auth in favor of SSH keys, and change default port."
    },
    "DoS slowloris": {
        "severity": "High",
        "color": "#f97316", # orange
        "description": "Low-bandwidth DoS holding HTTP connections open indefinitely by sending partial headers.",
        "mitigation": "Deploy reverse proxy with minimum incoming data rate check and connection timeout."
    },
    "DoS Slowhttptest": {
        "severity": "High",
        "color": "#f97316", # orange
        "description": "Slowloris variant testing slow HTTP headers, read, and post bodies against server endpoints.",
        "mitigation": "Limit maximum request header time and tune server read timeout thresholds."
    },
    "Botnet": {
        "severity": "Critical",
        "color": "#b91c1c", # deep red
        "description": "Infected zombie client communicating with Command & Control (C2) server.",
        "mitigation": "Isolate host on VLAN, terminate suspicious socket handles, and analyze host memory."
    },
    "Web Attack - Brute Force": {
        "severity": "Medium",
        "color": "#f59e0b", # amber
        "description": "Credential stuffing or dictionary attack against web application login portals.",
        "mitigation": "Enforce CAPTCHA, Multi-Factor Authentication (MFA), and rate-limit HTTP POST endpoints."
    },
    "Web Attack - XSS": {
        "severity": "High",
        "color": "#f97316", # orange
        "description": "Cross-Site Scripting injecting malicious JavaScript to hijack user session cookies.",
        "mitigation": "Sanitize and encode all untrusted inputs, deploy Content Security Policy (CSP) headers."
    },
    "Infiltration": {
        "severity": "Critical",
        "color": "#991b1b", # dark red
        "description": "Advanced Persistent Threat (APT) lateral movement across internal network segments.",
        "mitigation": "Segment internal networks with micro-segmentation, revoke lateral credentials."
    },
    "Web Attack - Sql Injection": {
        "severity": "Critical",
        "color": "#dc2626", # red
        "description": "Malicious SQL query injection attempting to exfiltrate database records or bypass auth.",
        "mitigation": "Use parameterized queries (PreparedStatements) and ORMs; deploy Web Application Firewall (WAF)."
    },
    "Heartbleed": {
        "severity": "Critical",
        "color": "#7f1d1d", # darkest red
        "description": "OpenSSL Heartbeat extension buffer over-read vulnerability exposing server memory/keys.",
        "mitigation": "Upgrade OpenSSL to patched version immediately, revoke and reissue SSL certificates."
    }
}


def get_threat_info(attack_name: str) -> Dict[str, Any]:
    """Returns threat intelligence metadata for a predicted attack class."""
    return THREAT_INTELLIGENCE.get(
        attack_name,
        {
            "severity": "Medium",
            "color": "#6366f1",
            "description": f"Detected cyber anomaly: {attack_name}.",
            "mitigation": "Inspect firewall logs and isolate affected host."
        }
    )
