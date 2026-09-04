import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
# Optional hostname overrides for environments with restricted DNS resolution.
import socket as _socket

_DNS_OVERRIDES = {
    "bedrock-mantle.us-east-2.api.aws": "18.116.7.134",
    "api.openai.com": "162.159.140.245",
    "<your-azure-resource>.services.ai.azure.com": "20.62.58.5",
}

_orig_getaddrinfo = _socket.getaddrinfo
def _patched_getaddrinfo(host, port, *args, **kwargs):
    if host in _DNS_OVERRIDES:
        ip = _DNS_OVERRIDES[host]
        return _orig_getaddrinfo(ip, port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)

_socket.getaddrinfo = _patched_getaddrinfo
