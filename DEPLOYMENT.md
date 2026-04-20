# =================================================================
# File: DEPLOYMENT.md
# Project: The Self-Pruning Neural Network
# Description: System execution pathways
# =================================================================

# Deployment Matrix 

## Local Development Setup
1. Fork and securely sync to the system: `git clone ...`
2. Create standard environment wrapper `python -m venv venv && source venv/bin/activate` or `venv\Scripts\activate`
3. Execute `pip install -r requirements.txt` mapping PyTorch against valid CUDA acceleration headers.
4. Run standard unit tests locally confirming compilation: `pytest tests/ -v`

## Docker Build & Run (Containerized Prod)
The Dockerfile implements a staged environment mapping eliminating development bloat on runtime modules.
```bash
docker build -t self-pruning-net .
docker run -d -p 8000:8000 --name pruning-container self-pruning-net
```

## Environment Variables
- `MODEL_PATH`: Directory specifying physical checkpoint locations (Defaults: `experiments/model_lambda_0.0001.pt`)
- `CUDA_VISIBLE_DEVICES`: Force container execution boundaries binding specific GPU layers.

## Production Checklist
1. **Health Verification:** Validate `GET http://localhost:8000/health`. Should resolve `status: ok` immediately.
2. **Stat Logging:** Confirm standard output writes cleanly.
3. **Target Validation:** Verify model checkpoint actively parses within lifespan without throwing 503 fallback headers.
4. **Latency Verification:** Invoke inference manually observing structural runtime limits compared to physical AI Agents requirements.

## Scaling Notes
Because inference processes strictly pruned sparse sub-nets, standard CPU scaling bounds scale linearly matching low-power requirements flawlessly without necessarily requiring strict hardware acceleration routing like massive cluster instances.
