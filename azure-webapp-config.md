# Azure App Service Deployment (Container)

Two services run inside one container:
- **FastAPI** on `$PORT` (Azure sets this env var).
- **Streamlit UI** on port `8501` (accessible via container port, not auto-exposed by App Service).

## Option A: Azure Web App for Containers

1. Build & push the image (e.g., to Azure Container Registry or Docker Hub):
   ```bash
   docker build -t <your-registry>/geography-nns:latest -f deploy/Dockerfile .
   docker push <your-registry>/geography-nns:latest
   ```

2. Create a Web App for Containers pointed at the image. No custom startup command needed
   (the container's `CMD` runs `startup.sh`).

3. Configure **Application Settings**:
   - `WEBSITES_PORT` = `8000`  (ensures health probing on FastAPI)
   - (Optional) `PORT` = `8000`

4. To access Streamlit from outside, add an **ingress/ingress route** or a separate Web App, or use an
   Application Gateway to route `/ui` → `8501`. For local dev, just `docker run -p 8000:8000 -p 8501:8501 ...`

## Option B: Azure App Service (Python)

If not using containers, set **Startup Command** to:
```
bash startup.sh
```
and deploy the repo. Make sure the App Service has build tools and `libgeos/proj` via a custom image or extension.
