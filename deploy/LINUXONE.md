# Running Zero Exposure on IBM LinuxONE (s390x)

Tested on the IBM LinuxONE Community Cloud (L1CC) for the IBM Z Datathon 2026: image `Ubuntu24.04-Datathon`, flavor `AI_LAB`
(3 vCPU, 8 GB RAM, 50 GB disk), event code `SSF2026`. The machine is an IBM z17 (type 9175, Telum II AI accelerator).

## 1. Prepare the VM (once)
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker "$USER"            # log out and in again, or keep using sudo for docker
git clone https://github.com/RitikBhardwaj-programmer/zero-exposure.git && cd zero-exposure
mkdir -p data/heart_disease && cd data/heart_disease
for h in cleveland hungarian switzerland va; do
  curl -fsSLO https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.$h.data
done
chmod a+r *.data && cd ../..               # the containers run as uid 1000
```

## 2. Build and start
```bash
time sudo docker compose -f deploy/docker-compose.yml build   # first build compiles grpcio + cryptography (~20 min on 3 vCPU)
sudo docker compose -f deploy/docker-compose.yml up -d
sudo docker compose -f deploy/docker-compose.yml ps -a        # certs Exited (0); superlink, 4 supernodes, dashboard Up
```
If the hospitals never join (a TLS problem), the plaintext fallback is one flag away (say "TLS off" in the demo):
`sudo docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.insecure.yml up -d`.

## 3. Open the dashboard from your laptop
The dashboard listens on the VM's loopback only (it has no login). Tunnel it over SSH and keep that window open:
```bash
ssh -i <your key> -L 8800:127.0.0.1:8000 linux1@<vm ip>
```
Then open http://127.0.0.1:8800. (Local port 8800, not 8000: Windows often reserves 7911-8010 for Hyper-V / Docker, and
`ssh -L 8000:...` then fails with `bind ... Permission denied`.)

## 4. Proof for the judges
```bash
uname -m; lscpu | grep -o nnpa | head -1                       # s390x, and the Telum AI accelerator flag
C="sudo docker compose -f deploy/docker-compose.yml"
$C exec -T superlink python -c "import sys, platform; print(platform.machine(), sys.byteorder)"   # s390x big
for i in 0 1 2 3; do $C exec -T supernode-$i ls /data /tls; done   # each hospital: only its own file, only ca.crt
$C logs | grep -cE "Traceback|ClientApp raised"                  # 0
```
A run's inspector files are written big-endian on s390x (`'descr': '>u4'` in `masked.npy`); the dashboard reads either order.

## 5. Read results correctly
On s390x, Python processes that loaded Flower can crash with a segmentation fault **when they exit**, after their work is done
(the pure-Python protobuf the image selects there). The dashboard therefore judges a run by its events file: all rounds plus
a final `done` event = success, whatever the exit code says.

## 6. Problems and fixes
| Symptom | Cause | Fix |
|---|---|---|
| pip read timeout during the build | slow mirror | the Dockerfile sets `PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10`; re-run the build (layers are cached) |
| `ImportError: libopenblas.so.0` on `import torch` | the s390x torch wheel links OpenBLAS | `libopenblas0` is in the apt list |
| Segfault on `import flwr` | protobuf's compiled extension on s390x | the image selects the pure-Python implementation on s390x only |
| A hospital "drops out"; SecAgg+ logs "halted" | an exception in a client (e.g. a byte-order error) | arrays are converted to native byte order on arrival (`zero_exposure/model.py`); check `docker compose logs supernode-N` |
| `service "certs" didn't complete successfully` | certificate maker failed | `docker compose logs certs` |
