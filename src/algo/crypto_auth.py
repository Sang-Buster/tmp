"""
Cryptographic authentication for MAVLink messages.

Supports multiple algorithms for performance comparison:
  - HMAC-SHA256: Standard MAVLink v2 signing (fastest, MAC only)
  - ChaCha20-Poly1305: AEAD stream cipher (fast, encrypts + authenticates)
  - AES-256-CTR + HMAC: Block cipher with separate MAC (traditional, heavier)

When enabled, only messages with valid signatures are accepted; spoofed/modified
messages are rejected — neutralizing phantom, falsification, and coordinate attacks.
"""
import hashlib
import hmac
import json
import os
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .mavlink import MAVLinkMessage


class CryptoAlgorithm(str, Enum):
    HMAC_SHA256 = "hmac_sha256"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    AES_256_CTR = "aes_256_ctr"


ALGORITHM_LABELS = {
    CryptoAlgorithm.HMAC_SHA256: "HMAC-SHA256",
    CryptoAlgorithm.CHACHA20_POLY1305: "ChaCha20-Poly1305",
    CryptoAlgorithm.AES_256_CTR: "AES-256-CTR",
}


@dataclass
class CryptoStats:
    accepted: int = 0
    rejected: int = 0
    sign_time_us: float = 0.0
    verify_time_us: float = 0.0
    rejection_log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        total = max(1, self.accepted + self.rejected)
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejected / total, 4),
            "avg_sign_time_us": round(self.sign_time_us / max(1, self.accepted + self.rejected), 2),
            "avg_verify_time_us": round(self.verify_time_us / max(1, self.accepted + self.rejected), 2),
            "recent_rejections": self.rejection_log[-10:],
        }

    def reset(self):
        self.accepted = 0
        self.rejected = 0
        self.sign_time_us = 0.0
        self.verify_time_us = 0.0
        self.rejection_log.clear()


class CryptoAuth:
    """
    Multi-algorithm message authentication for MAVLink.

    Each real agent gets a pre-shared 32-byte key. The signing/verification
    algorithm can be switched at runtime for performance comparison.
    """

    def __init__(self):
        self.enabled: bool = False
        self.algorithm: CryptoAlgorithm = CryptoAlgorithm.HMAC_SHA256
        self._agent_keys: dict[str, bytes] = {}
        self.stats = CryptoStats()
        self._nonce_counters: dict[str, int] = {}

    def set_algorithm(self, algo: str):
        try:
            self.algorithm = CryptoAlgorithm(algo)
        except ValueError:
            self.algorithm = CryptoAlgorithm.HMAC_SHA256

    def generate_keys(self, agent_ids: list[str]):
        """Generate fresh 32-byte keys for all real agents."""
        self._agent_keys.clear()
        self._nonce_counters.clear()
        for aid in agent_ids:
            self._agent_keys[aid] = os.urandom(32)
            self._nonce_counters[aid] = 0

    def add_agent_key(self, agent_id: str):
        """Add a key for a single new agent without disrupting existing keys."""
        if agent_id not in self._agent_keys:
            self._agent_keys[agent_id] = os.urandom(32)
            self._nonce_counters[agent_id] = 0

    def remove_agent_key(self, agent_id: str):
        """Remove an agent's key when the agent is deleted."""
        self._agent_keys.pop(agent_id, None)
        self._nonce_counters.pop(agent_id, None)

    def has_key(self, agent_id: str) -> bool:
        return agent_id in self._agent_keys

    def sign_message(self, msg: MAVLinkMessage) -> MAVLinkMessage:
        """Sign a message using the sender's key with the selected algorithm."""
        key = self._agent_keys.get(msg.sender_id)
        if key is None:
            return msg

        t0 = time.perf_counter_ns()
        payload_bytes = self._serialize_payload(msg)

        if self.algorithm == CryptoAlgorithm.HMAC_SHA256:
            msg.signature = self._sign_hmac(key, payload_bytes)
        elif self.algorithm == CryptoAlgorithm.CHACHA20_POLY1305:
            msg.signature = self._sign_chacha20(key, payload_bytes, msg.sender_id)
        elif self.algorithm == CryptoAlgorithm.AES_256_CTR:
            msg.signature = self._sign_aes_ctr(key, payload_bytes, msg.sender_id)

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self.stats.sign_time_us += elapsed_us
        return msg

    def verify_message(self, msg: MAVLinkMessage) -> bool:
        if not self.enabled:
            return True

        key = self._agent_keys.get(msg.sender_id)
        if key is None:
            return False
        if msg.signature is None:
            return False

        t0 = time.perf_counter_ns()
        payload_bytes = self._serialize_payload(msg)

        if self.algorithm == CryptoAlgorithm.HMAC_SHA256:
            valid = self._verify_hmac(key, payload_bytes, msg.signature)
        elif self.algorithm == CryptoAlgorithm.CHACHA20_POLY1305:
            valid = self._verify_chacha20(key, payload_bytes, msg.signature)
        elif self.algorithm == CryptoAlgorithm.AES_256_CTR:
            valid = self._verify_aes_ctr(key, payload_bytes, msg.signature)
        else:
            valid = False

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self.stats.verify_time_us += elapsed_us
        return valid

    def filter_messages(self, messages: list[MAVLinkMessage]) -> list[MAVLinkMessage]:
        if not self.enabled:
            self.stats.accepted += len(messages)
            return messages

        accepted = []
        for msg in messages:
            if self.verify_message(msg):
                accepted.append(msg)
                self.stats.accepted += 1
            else:
                self.stats.rejected += 1
                self.stats.rejection_log.append({
                    "time": time.time(),
                    "sender": msg.sender_id,
                    "type": msg.msg_type.name,
                    "spoofed": msg.is_spoofed,
                    "algorithm": self.algorithm.value,
                })
        return accepted

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "algorithm": self.algorithm.value,
            "algorithm_label": ALGORITHM_LABELS.get(self.algorithm, self.algorithm.value),
            "registered_agents": list(self._agent_keys.keys()),
            "stats": self.stats.to_dict(),
        }

    def reset(self):
        self._agent_keys.clear()
        self._nonce_counters.clear()
        self.stats.reset()

    # ----------------------------------------------------------------
    # HMAC-SHA256 (MAC-only, fastest)
    # ----------------------------------------------------------------

    @staticmethod
    def _sign_hmac(key: bytes, data: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha256).digest()

    @staticmethod
    def _verify_hmac(key: bytes, data: bytes, signature: bytes) -> bool:
        expected = hmac.new(key, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)

    # ----------------------------------------------------------------
    # ChaCha20-Poly1305 (AEAD — encrypt + authenticate)
    # ----------------------------------------------------------------

    def _sign_chacha20(self, key: bytes, data: bytes, sender_id: str) -> bytes:
        """
        ChaCha20-Poly1305 AEAD: encrypts the payload and produces a 16-byte
        authentication tag. We store nonce + ciphertext + tag as the signature.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError:
            return self._sign_hmac(key, data)

        nonce = self._next_nonce(sender_id)
        cipher = ChaCha20Poly1305(key)
        ct = cipher.encrypt(nonce, data, None)
        return nonce + ct  # 12-byte nonce + ciphertext + 16-byte tag

    def _verify_chacha20(self, key: bytes, data: bytes, signature: bytes) -> bool:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError:
            return self._verify_hmac(key, data, signature)

        if len(signature) < 12:
            return False
        nonce = signature[:12]
        ct = signature[12:]
        cipher = ChaCha20Poly1305(key)
        try:
            plaintext = cipher.decrypt(nonce, ct, None)
            return plaintext == data
        except Exception:
            return False

    # ----------------------------------------------------------------
    # AES-256-CTR + HMAC-SHA256 (encrypt-then-MAC)
    # ----------------------------------------------------------------

    def _sign_aes_ctr(self, key: bytes, data: bytes, sender_id: str) -> bytes:
        """
        AES-256-CTR encryption then HMAC-SHA256 over the ciphertext.
        Signature = iv(16) + ciphertext + hmac(32).
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError:
            return self._sign_hmac(key, data)

        iv = os.urandom(16)
        enc_key = key[:16] + key[:16]  # expand to 32 bytes for AES-256
        mac_key = key[16:] + key[:16]

        cipher = Cipher(algorithms.AES(enc_key), modes.CTR(iv))
        encryptor = cipher.encryptor()
        ct = encryptor.update(data) + encryptor.finalize()

        mac = hmac.new(mac_key, iv + ct, hashlib.sha256).digest()
        return iv + ct + mac

    def _verify_aes_ctr(self, key: bytes, data: bytes, signature: bytes) -> bool:
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError:
            return self._verify_hmac(key, data, signature)

        if len(signature) < 48:  # 16 (iv) + min 1 (ct) + 32 (mac)
            return False

        mac_len = 32
        iv = signature[:16]
        ct = signature[16:-mac_len]
        received_mac = signature[-mac_len:]

        enc_key = key[:16] + key[:16]
        mac_key = key[16:] + key[:16]

        expected_mac = hmac.new(mac_key, iv + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(received_mac, expected_mac):
            return False

        cipher = Cipher(algorithms.AES(enc_key), modes.CTR(iv))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ct) + decryptor.finalize()
        return plaintext == data

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _next_nonce(self, sender_id: str) -> bytes:
        """Generate an incrementing 12-byte nonce per sender."""
        counter = self._nonce_counters.get(sender_id, 0)
        self._nonce_counters[sender_id] = counter + 1
        return struct.pack("<Q", counter) + os.urandom(4)

    @staticmethod
    def _serialize_payload(msg: MAVLinkMessage) -> bytes:
        canonical = {
            "msg_type": int(msg.msg_type),
            "sender_id": msg.sender_id,
            "sequence": msg.sequence,
            "payload": msg.payload,
        }
        return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


# Singleton
_crypto_auth: Optional[CryptoAuth] = None


def get_crypto_auth() -> CryptoAuth:
    global _crypto_auth
    if _crypto_auth is None:
        _crypto_auth = CryptoAuth()
    return _crypto_auth


def reset_crypto_auth():
    global _crypto_auth
    if _crypto_auth is not None:
        _crypto_auth.reset()
    _crypto_auth = None
