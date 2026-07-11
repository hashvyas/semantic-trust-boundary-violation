"""
pki/pki_layer.py
==================
PKI / SCMS authentication layer -- FIRST stage of the frozen V2X Trust
Stack (Incoming Message -> PKI -> B1 -> MBD -> B2 -> CP -> B3 ->
TrustDecisionEngine -> Adapters -> DS MASS -> Dispatcher).

Integrated from the uploaded modules/pki.py. Two changes made during
integration, both purely structural (no algorithm changes):

1. `"boundary"` renamed from `"B1_PKI"` to `"PKI"` -- resolves the
   label collision identified in the responsibility audit (§0): "B1" is
   reserved exclusively for this repo's validated cryptographic-trust-
   assessment layer, which is a DIFFERENT thing from this PKI layer.
   PKI answers "is this cert/signature valid"; B1 answers "given PKI's
   verdict plus structural/protocol checks, what's the crypto trust
   assessment". PKI feeds B1, PKI is not B1.
2. Added `revoked` and `compromised` as explicit top-level fields on the
   result (previously only visible by inspecting the certificate dict) --
   audit finding D1 requires MBD to see the "compromised but not yet
   revoked" state without reaching into PKI's internal certificate
   representation.

Everything else (RSA-PSS sign/verify, CertificateAuthority, expiry
logic) is unchanged from the uploaded implementation.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# --------------------------------------------------
# Certificate Authority (in-memory, for simulation)
# --------------------------------------------------
class CertificateAuthority:
    """Mimics a (very) simplified SCMS-style CA. Issues key pairs + certs,
    tracks revocation and the "compromised but not yet revoked" state
    audit finding D1 requires (a cert can be 100% cryptographically valid
    while still being used maliciously -- PKI is intentionally "blind" to
    that; MBD/B2/B3 are where it becomes visible)."""

    def __init__(self) -> None:
        self._certs: Dict[str, dict] = {}
        self._compromised: set = set()

    def issue_certificate(self, sender_id: str, validity_days: int = 365):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()

        cert = {
            "cert_id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "issued_at": datetime.datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
            ).isoformat(),
            "revoked": False,
        }
        self._certs[sender_id] = cert
        return private_key, public_key, cert

    def revoke(self, sender_id: str) -> None:
        if sender_id in self._certs:
            self._certs[sender_id]["revoked"] = True

    def mark_compromised(self, sender_id: str) -> None:
        """Flags a cert as compromised WITHOUT revoking it -- this is the
        deliberate PKI-blind-spot state the STBV threat model depends on."""
        self._compromised.add(sender_id)

    def is_compromised(self, sender_id: str) -> bool:
        return sender_id in self._compromised

    def is_valid(self, cert: dict) -> bool:
        if cert.get("revoked"):
            return False
        expiry = datetime.datetime.fromisoformat(cert["expires_at"])
        return datetime.datetime.utcnow() <= expiry


# --------------------------------------------------
# Sign / Verify (unchanged from uploaded implementation)
# --------------------------------------------------
def sign_message(msg: dict, private_key) -> bytes:
    message_bytes = json.dumps(msg, sort_keys=True).encode()
    signature = private_key.sign(
        message_bytes,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return signature


def verify_signature(msg: dict, signature: bytes, public_key) -> bool:
    message_bytes = json.dumps(msg, sort_keys=True).encode()
    try:
        public_key.verify(
            signature,
            message_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# --------------------------------------------------
# Full PKI Layer
# --------------------------------------------------
def pki_layer(
    msg: dict,
    signature: bytes,
    certificate: dict,
    public_key,
    ca: CertificateAuthority,
) -> Dict[str, Any]:
    """Returns the PKI verdict -- the ONLY thing B1 is allowed to see.
    B1 never sees the raw signature or key material, only this dict.
    That hand-off (verdict only, no key material) is exactly the
    trust-boundary behavior the STBV paper depends on.
    """
    cert_valid = ca.is_valid(certificate)
    sig_valid = verify_signature(msg, signature, public_key)
    sender_id = certificate["sender_id"]

    result = {
        "boundary": "PKI",  # renamed from "B1_PKI" -- see module docstring
        "sender": sender_id,
        "cert_id": certificate.get("cert_id"),
        "cert_valid": cert_valid,
        "sig_valid": sig_valid,
        "pki_pass": cert_valid and sig_valid,
        "revoked": bool(certificate.get("revoked", False)),
        "compromised": ca.is_compromised(sender_id),
    }
    return result
