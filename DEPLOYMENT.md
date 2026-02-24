# SAM 3D Body Deployment Guide

This guide covers deploying the SAM 3D Body inference service for production use.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3070 or better recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for model checkpoints and temporary files
- **Network**: High-bandwidth connection for video downloads/uploads

### Software Requirements

- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **CUDA**: 11.8 or 12.1
- **Python**: 3.10 or 3.11
- **Docker**: (optional) for containerized deployment

## Installation

### 1. Clone and Setup

```bash
cd /opt
sudo mkdir sam-3d-body
sudo chown $USER:$USER sam-3d-body
cd sam-3d-body

# Clone repository
git clone https://github.com/your-org/duolian-pose.git .
cd sam-3d-body
```

### 2. Create Python Environment

```bash
# Install base dependencies
conda create -n sam3db python=3.11 -y
conda activate sam3db

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
pip install -r requirements-api.txt

# Install base SAM 3D Body dependencies (see INSTALL.md)
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

### 3. Download Model Checkpoints

```bash
# Install huggingface-cli
pip install huggingface-hub

# Login (required for gated models)
huggingface-cli login

# Download checkpoints
hf download facebook/sam-3d-body-dinov3 \
  --local-dir ./checkpoints/sam-3d-body-dinov3

# Verify checkpoints
ls -la ./checkpoints/sam-3d-body-dinov3/
# Should contain: model.ckpt, assets/mhr_model.pt, model_config.yaml
```

### 4. Configuration

Create environment file:

```bash
cat > .env << EOF
# API Configuration
SAM3DB_API_HOST=0.0.0.0
SAM3DB_API_PORT=8000
SAM3DB_API_WORKERS=1

# Model Configuration
SAM3DB_MODEL_CHECKPOINT_PATH=./checkpoints/sam-3d-body-dinov3/model.ckpt
SAM3DB_MHR_MODEL_PATH=./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
SAM3DB_DETECTOR_NAME=vitdet
SAM3DB_DEVICE=cuda

# Processing Configuration
SAM3DB_DEFAULT_TARGET_FPS=30.0
SAM3DB_MAX_VIDEO_DURATION=30.0
SAM3DB_MAX_VIDEO_FILE_SIZE_MB=500
SAM3DB_TEMP_DIR=/tmp/sam3db
SAM3DB_OUTPUT_DIR=./output

# Feature Flags
SAM3DB_ENABLE_SMASH_ANALYSIS=true
SAM3DB_SUPPORT_LEFT_HANDED=false
SAM3DB_SUPPORT_MULTI_CAMERA=false
SAM3DB_ENABLE_PHASE_DETECTION=false
EOF
```

### 5. Create Directories

```bash
mkdir -p /tmp/sam3db
mkdir -p ./output
chmod 755 /tmp/sam3db
```

## Running the Service

### Development Mode

```bash
conda activate sam3db
python scripts/start_server.py --reload
```

### Production Mode

```bash
conda activate sam3db
python scripts/start_server.py --port 8000 --workers 1
```

### Using Systemd

Create service file:

```bash
sudo tee /etc/systemd/system/sam-3d-body.service << 'EOF'
[Unit]
Description=SAM 3D Body Inference Service
After=network.target

[Service]
Type=simple
User=sam3db
Group=sam3db
WorkingDirectory=/opt/sam-3d-body
Environment="PATH=/home/sam3db/anaconda3/envs/sam3db/bin"
Environment="SAM3DB_API_PORT=8000"
Environment="SAM3DB_DEVICE=cuda"
Environment="SAM3DB_MODEL_CHECKPOINT_PATH=/opt/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
ExecStart=/home/sam3db/anaconda3/envs/sam3db/bin/python scripts/start_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create user
sudo useradd -r -s /bin/false sam3db
sudo usermod -aG video sam3db

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable sam-3d-body
sudo systemctl start sam-3d-body

# Check status
sudo systemctl status sam-3d-body
sudo journalctl -u sam-3d-body -f
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t sam-3d-body:latest .
```

### Run Docker Container

```bash
docker run -d \
  --name sam-3d-body \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v /tmp/sam3db:/tmp/sam3db \
  -e SAM3DB_API_HOST=0.0.0.0 \
  -e SAM3DB_DEVICE=cuda \
  sam-3d-body:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  sam-3d-body:
    build: .
    image: sam-3d-body:latest
    container_name: sam-3d-body
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - SAM3DB_API_HOST=0.0.0.0
      - SAM3DB_API_PORT=8000
      - SAM3DB_DEVICE=cuda
      - SAM3DB_MODEL_CHECKPOINT_PATH=/app/checkpoints/sam-3d-body-dinov3/model.ckpt
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./output:/app/output
      - /tmp/sam3db:/tmp/sam3db
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Health Checks

### API Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-24T10:30:00",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_mb": 24576,
  "model_loaded": true,
  "model_version": "sam-3d-body-1.0.0"
}
```

### GPU Check

```bash
nvidia-smi
```

Should show Python process using GPU memory.

## Monitoring

### Prometheus Metrics (if enabled)

```bash
curl http://localhost:8000/metrics
```

### Logs

```bash
# Systemd
sudo journalctl -u sam-3d-body -f

# Docker
docker logs -f sam-3d-body

# Direct
conda activate sam3db
python scripts/start_server.py 2>&1 | tee server.log
```

## Load Balancing

For high-throughput deployments, use a reverse proxy:

### Nginx

```nginx
upstream sam3d_backend {
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://sam3d_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

## Security

### Firewall

```bash
# Allow only backend servers
sudo ufw allow from 10.0.0.0/8 to any port 8000
sudo ufw deny 8000
```

### TLS/SSL

Use a reverse proxy (nginx, traefik) for TLS termination.

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Check permissions
sudo usermod -aG video $USER
# Logout and login again
```

### Out of Memory

- Reduce `SAM3DB_MAX_VIDEO_DURATION`
- Process videos sequentially (not parallel)
- Use smaller batch sizes
- Monitor with `nvidia-smi dmon`

### Slow Inference

- Ensure GPU is being used (check `nvidia-smi`)
- Use SSD for temp directory
- Check network bandwidth for video downloads
- Consider using local file paths instead of URLs

### Model Loading Fails

```bash
# Verify checkpoints
ls -la checkpoints/sam-3d-body-dinov3/

# Check file permissions
file checkpoints/sam-3d-body-dinov3/model.ckpt

# Re-download if corrupted
rm -rf checkpoints/sam-3d-body-dinov3
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

## Performance Tuning

### GPU Optimization

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Set power limit (adjust for your GPU)
sudo nvidia-smi -pl 250
```

### System Optimization

```bash
# Increase file descriptor limits
echo "sam3db soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "sam3db hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Tune network
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

## Backup and Recovery

### Checkpoint Backup

```bash
# Backup checkpoints
rsync -av checkpoints/ /backup/sam-3d-body/checkpoints/

# Or to S3
aws s3 sync checkpoints/ s3://my-bucket/sam-3d-body-checkpoints/
```

### Service Recovery

```bash
# Restart service
sudo systemctl restart sam-3d-body

# Check logs
sudo journalctl -u sam-3d-body --since "1 hour ago"
```

## Integration with cf-backend

The cf-backend worker will call this service via HTTP. Ensure:

1. **Network connectivity** between Cloudflare Worker and GPU service
2. **Firewall rules** allow Worker IPs (or use authenticated tunnel)
3. **Health endpoint** is accessible for load balancer checks
4. **Request timeout** in worker is >= service timeout (300s)

### Worker Configuration

In cf-backend, set environment variable:

```bash
SAM3D_BODY_API_URL=https://gpu-inference.example.com
```

## Support

For issues:
1. Check logs: `journalctl -u sam-3d-body -f`
2. Verify GPU: `nvidia-smi`
3. Test health endpoint: `curl localhost:8000/health`
4. Check protocol compliance: See `cf-backend/docs/pose-correction-protocol.md`
