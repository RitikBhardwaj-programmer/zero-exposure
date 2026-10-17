"""Make a demo certificate authority (CA) and the SuperLink's TLS certificate.

The SuperLink gets server.pem + server.key (+ ca.crt); hospitals (SuperNodes), the flwr CLI and the dashboard get ONLY ca.crt,
which lets them check they talk to the real SuperLink. ca.key signs certificates and never leaves the machine that made it.
Names in the certificate: `superlink` (Docker), `localhost`, 127.0.0.1 and ::1 (Flower's own ServerApp process dials 127.0.0.1).

Use:  python deploy/gen_certs.py [--out DIR] [--force]      (default DIR: deploy/certs; existing files are kept unless --force)
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

DEFAULT_OUT = Path(__file__).resolve().parent / "certs"
DNS_NAMES = ["superlink", "localhost"]
IPS = ["127.0.0.1", "::1"]
DAYS = 365


def _write_key(path: Path, key) -> None:
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def generate(out: Path = DEFAULT_OUT, force: bool = False) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = [out / n for n in ("ca.crt", "ca.key", "server.pem", "server.key")]
    if all(f.exists() for f in files) and not force:
        return out
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Zero Exposure demo CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=DAYS))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                                       encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    san = [x509.DNSName(n) for n in DNS_NAMES] + [x509.IPAddress(ipaddress.ip_address(i)) for i in IPS]
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "superlink")]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=DAYS))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    (out / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    _write_key(out / "ca.key", ca_key)
    (out / "server.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_key(out / "server.key", key)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"certificates in {generate(Path(a.out), a.force)}")
