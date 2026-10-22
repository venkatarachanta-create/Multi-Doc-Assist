# Deploying Multi-Document Assist to Azure

Target: **Azure App Service — Web App for Containers**. One container, one public
URL, closest match to the Render setup you already have — good for a 3-hour
teaching session, no Kubernetes concepts required.

Two ways to get there, used in this order across Week 7 → Week 8:

1. **Manual, with `deploy.sh`** — Week 8 Part 1. You *see* every resource get
   created, in order, so the pieces (registry, plan, web app, app settings)
   stay concrete instead of hidden behind a pipeline.
2. **CI/CD, with GitHub Actions** — Week 8 Part 2. Once the manual path works,
   wire up `azure-deploy.yml` so every push to `main` redeploys automatically.

---

## Prerequisites

- An Azure subscription ([free tier](https://azure.microsoft.com/free/) is enough for a B1 plan)
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed, then `az login`
- Your `OPENAI_API_KEY`, `COHERE_API_KEY`, and `OPENAI_MODEL` (e.g. `gpt-4o-mini`)
  — `OPENAI_MODEL` is easy to forget since it's not a secret, but `llm.py` reads
  it directly and the app 400s ("you must provide a model parameter") without it
- **Docker Desktop, if you're on Azure for Students or another free/trial
  subscription** — `az acr build` (cloud-side build, no local Docker needed) is
  blocked on those with `TasksOperationsNotAllowed`, an anti-abuse measure. See
  the troubleshooting section below; the short version is `docker build` +
  `docker push` locally instead of step 3.

## 1. Manual deploy

```bash
cd deploy/azure
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBEDDING_MODEL="..."         # optional
# optional, only if you want LangSmith tracing from the deployed app:
export LANGSMITH_API_KEY="..."
export LANGCHAIN_PROJECT="week7-rag-azure"

chmod +x deploy.sh
./deploy.sh
```

What it does, step by step (matches the numbered `echo` lines in the script):

| # | Command | What it creates |
|---|---------|------------------|
| 1 | `az group create` | A resource group — everything below lives inside it, so cleanup is one command |
| 2 | `az acr create` | Azure Container Registry — a private place to store the image |
| 3 | `az acr build` | Builds the image from `../Dockerfile` **on Azure's build agents**, pushes it to ACR |
| 4 | `az appservice plan create` | The VM capacity (Linux, B1 tier) the web app runs on |
| 5 | `az webapp create` | The Web App itself, pointed at the image in ACR |
| 6 | `az webapp config container set` | Gives the Web App permission to pull from ACR |
| 7 | `az webapp config appsettings set` | Sets `WEBSITES_PORT=8080` + your API keys and model names as app settings (Azure's equivalent of env vars — never baked into the image) |
| 8 | — | Prints the URL and the log-tail command |

When it finishes:

```bash
az webapp log tail --resource-group rg-multidoc-assist --name <your-app-name>
```

Open `https://<your-app-name>.azurewebsites.net` — first load takes 1-2
minutes while the container pulls and FastAPI + Streamlit both boot inside it
(the same `entrypoint.sh` sequencing you already use locally and on Render).

### Why these specific settings

- **`WEBSITES_PORT=8080`** — Azure doesn't set `$PORT` the way Render does. You
  tell it which port your container listens on instead. `entrypoint.sh`
  defaults to 8080 when `$PORT` isn't set — deliberately *not* 8000, since
  that's FastAPI's fixed internal port and a `0.0.0.0` bind collides with a
  `127.0.0.1` bind on the same port number (Streamlit fails with "port already
  in use" if these match).
- **`OPENAI_MODEL` as an app setting, not just a secret** — it's not sensitive,
  but `llm.py` reads `settings.OPENAI_MODEL` directly with no fallback, so a
  missing value makes the app 400 on every question, not fail to start. Easy
  to forget because it "isn't a secret" — forward it anyway.
- **API keys as app settings, not baked into the image** — same reason
  `.env` isn't in the Dockerfile: rotate a key without rebuilding, and it
  never ends up sitting in an image layer someone can `docker history` open.
- **ACR admin credentials** — the fastest path for a course. In a real
  production setup you'd use a managed identity instead (`az webapp identity
  assign` + `az role assignment create` for `AcrPull`) so no credential is
  stored anywhere at all — worth mentioning to students, not worth the extra
  15 minutes of setup live.

## 2. CI/CD with GitHub Actions

`.github/workflows/azure-deploy.yml` builds the image and redeploys the Web
App on every push to `main`. To wire it up against the resources `deploy.sh`
just created:

1. **Repo variables** (Settings → Secrets and variables → Actions → *Variables* tab):
   - `AZURE_WEBAPP_NAME` — the `$APP_NAME` deploy.sh printed
   - `ACR_NAME` — the `$ACR_NAME` you used (default `acrmultidocassist`)
2. **Repo secrets** (same page, *Secrets* tab):
   - `ACR_USERNAME`, `ACR_PASSWORD` — from `az acr credential show --name <ACR_NAME>`
   - `AZURE_CREDENTIALS` — a service principal with rights to deploy:
     ```bash
     az ad sp create-for-rbac \
       --name "sp-multidoc-assist-ci" \
       --role contributor \
       --scopes /subscriptions/<subscription-id>/resourceGroups/rg-multidoc-assist \
       --sdk-auth
     ```
     Paste the full JSON output as the `AZURE_CREDENTIALS` secret.
3. Push to `main`. Check the **Actions** tab — build, push to ACR, then
   `azure/webapps-deploy` swaps the Web App to the new image tag (the git SHA).

This is the "traceability" habit from Week 6-7 applied to infrastructure: every
deployed image is tagged with the commit that produced it, so `az webapp show`
tells you exactly what code is live.

## Monitoring & logs

```bash
# live tail, same idea as watching LangSmith traces but for the container itself
az webapp log tail --resource-group rg-multidoc-assist --name <app-name>

# quick health check from your terminal
curl https://<app-name>.azurewebsites.net/_stcore/health
```

For anything beyond log tailing (request latency, failure rate over time),
enable **Application Insights** from the Azure Portal on the Web App — one
toggle, no code changes required for basic HTTP metrics.

## Known limitation: storage is ephemeral

The container's filesystem (uploaded PDFs, the FAISS index) does **not**
persist across restarts or scale events on App Service by default — every
redeploy starts clean. That's fine for a teaching demo; for a real deployment,
mount **Azure Files** as a persistent volume (`az webapp config storage-account
add`) or move the index into a managed vector store. Flag this explicitly to
students — it's the same "honest limitations" habit from the Week 6 deck,
applied to this project.

## Cleanup

```bash
cd deploy/azure
./teardown.sh
```

Deletes the whole resource group (registry, plan, web app — everything from
`deploy.sh`) so a B1 plan doesn't sit there billing between sessions.

## Troubleshooting

Real issues hit while first running this deploy, in the order you'd hit them:

| Symptom | Cause | Fix |
|---|---|---|
| `az acr build` fails with `TasksOperationsNotAllowed` | ACR Tasks (cloud-side builds) are blocked on Azure for Students / free-trial subscriptions | Build locally instead: `az acr login --name <ACR_NAME>`, `docker build -t <ACR_NAME>.azurecr.io/multidoc-assist:v1 ..`, `docker push ...`, then continue at step 4 |
| `az acr build` / `az acr login` errors with `Unable to find 'Dockerfile'` | `--file` is resolved relative to your shell's **cwd**, not the source-context argument | Run the command with `Live/` as cwd and pass `--file Dockerfile .` (context `.`, not `..`) |
| Random `ConnectionResetError` / `Connection aborted` on various `az` commands | Transient network blips between the CLI and Azure's management endpoints | Just retry — most `az` write commands are idempotent (`az group create`, `az acr create`, `az webapp create` all succeed silently if already created) |
| `az webapp config container set` / `az webapp config appsettings set|list` **fail every single time** with the same `ConnectionResetError`, traceback ending in `is_flex_functionapp` → `send_raw_request` | A reproducible bug in some Azure CLI versions — these commands do an extra internal check that fails on some networks even after `az upgrade` | Do these two steps in the **Azure Portal** instead: Deployment Center (image/registry) and Environment variables (app settings). Also sidesteps ever pasting API keys into a terminal. |
| Site returns 503 indefinitely (not just a slow cold start) | `WEBSITES_PORT` matched FastAPI's internal port (8000), so Streamlit couldn't bind — see `entrypoint.sh` header comment | Set `WEBSITES_PORT=8080` (already the default in this repo) and rebuild if you'd hardcoded 8000 anywhere |
| App loads, but every question fails with `openai.BadRequestError: you must provide a model parameter` | `OPENAI_MODEL` app setting missing — `llm.py` has no fallback | Add `OPENAI_MODEL` (e.g. `gpt-4o-mini`) as an app setting |
| Config change in the Portal doesn't seem to take effect | Log stream can show **stale, buffered** output from before the change — identical timestamps to a previous failure is the tell | Explicitly click **Restart** on the Overview page after saving config changes, don't rely on auto-restart |

## Optional: infrastructure as code

`azure-webapp.bicep` in this folder defines the same resources (ACR, plan, web
app, app settings) declaratively, for students who want to see the
Infrastructure-as-Code version instead of imperative `az` commands. Not part
of the live-session critical path — point to it as a "if you want to go
further" reference.
