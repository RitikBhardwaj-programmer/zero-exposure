# One image for every role (SuperLink, SuperNode, dashboard, certificate maker). Builds for linux/amd64 and linux/s390x (IBM LinuxONE).
# Build from the repo root:  docker compose -f deploy/docker-compose.yml build
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PYTHONUNBUFFERED=1 \
    FLWR_TELEMETRY_ENABLED=0 FLWR_DISABLE_UPDATE_CHECK=1

# numpy / scipy / pandas / scikit-learn from apt: PyPI has no s390x wheels for them, Ubuntu ships s390x builds.
# Compiler, Rust and -dev headers: pip must build grpcio and cryptography from source on s390x.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip python3-numpy python3-scipy python3-pandas python3-sklearn \
        build-essential pkg-config libssl-dev libffi-dev zlib1g-dev libc-ares-dev rustc cargo ca-certificates libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# --system-site-packages: the venv uses the apt scientific stack instead of rebuilding it.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch: CPU wheels (s390x included) live on the PyTorch index, not on PyPI. Its own layer: it is the second-largest download.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# grpcio compiles for a long time on s390x (use the system OpenSSL / zlib / c-ares). Own layer, so later changes do not redo it.
ENV GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 GRPC_PYTHON_BUILD_SYSTEM_CARES=1 GRPC_PYTHON_BUILD_EXT_COMPILER_JOBS=4
RUN pip install --no-cache-dir "grpcio>=1.70" "grpcio-health-checking>=1.70" cryptography

RUN pip install --no-cache-dir flwr==1.37.0 opacus==1.6.0 fastapi uvicorn

# s390x: protobuf's compiled extension crashed on import there (seen under emulation); the pure-Python one is correct on any
# byte order. A .pth line runs at every Python start and selects it only on s390x.
RUN if [ "$(uname -m)" = "s390x" ]; then \
        echo "import os; os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')" \
        > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/zz_protobuf_python.pth"; \
    fi

# Run as uid 1000 (the image's existing `ubuntu` user). Folders that named volumes mount over are created here, owned by it,
# so the volumes start out owned by it too.
RUN mkdir -p /app /runs /data /tls /certs-ca /certs-server /certs-public && chown -R 1000:1000 /app /runs /data /tls /certs-ca /certs-server /certs-public

WORKDIR /app
COPY --chown=1000:1000 . /app
USER 1000
