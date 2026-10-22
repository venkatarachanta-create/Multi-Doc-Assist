#!/bin/bash
# ---------------------------------------------------------------------------
# Deploy Multi-Document Assist to Azure App Service (Web App for Containers).
#
# Run this from the "Live" folder (it builds the image from ".."), with the
# Azure CLI installed and logged in (`az login`). Everything here is scoped
# to one resource group so `az group delete` at the end removes it all.
#
# Required in your shell before running:
#   export OPENAI_API_KEY="sk-..."
#   export COHERE_API_KEY="..."
#   export OPENAI_MODEL="gpt-4o-mini"       # required — llm.py 400s without it
#   export OPENAI_EMBEDDING_MODEL="..."     # optional
#   export LANGSMITH_API_KEY="..."          # optional, for tracing
#   export LANGCHAIN_PROJECT="week7-rag"    # optional, for tracing
#
# NOTE on step 3 (az acr build): this fails with "TasksOperationsNotAllowed" on
# Azure for Students / free-trial subscriptions (ACR Tasks are blocked there to
# prevent free-compute abuse). If you hit that, build+push locally instead:
#   az acr login --name <ACR_NAME>
#   docker build -t <ACR_NAME>.azurecr.io/multidoc-assist:v1 ..
#   docker push <ACR_NAME>.azurecr.io/multidoc-assist:v1
# then skip to step 4.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Config you can change -------------------------------------------------
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-multidoc-assist}"
LOCATION="${LOCATION:-centralindia}"
ACR_NAME="${ACR_NAME:-acrmultidocassist}"        # must be globally unique, letters+digits only
PLAN_NAME="${PLAN_NAME:-plan-multidoc-assist}"
APP_NAME="${APP_NAME:-multidoc-assist-$RANDOM}"  # must be globally unique (becomes *.azurewebsites.net)
IMAGE_NAME="multidoc-assist"
IMAGE_TAG="${IMAGE_TAG:-v1}"
SKU="${SKU:-B1}"                                  # B1 = cheapest always-on tier; F1 (free) can't run 2 processes reliably
# -----------------------------------------------------------------------------

echo "== 1. Resource group =="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output table

echo "== 2. Azure Container Registry =="
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true --output table

echo "== 3. Build the image *in* ACR (no local Docker needed) =="
# --file is resolved relative to the current shell's cwd, not the source-context
# argument — so this has to run with Live/ as cwd, and the context is then "."
(cd .. && az acr build \
  --registry "$ACR_NAME" \
  --image "${IMAGE_NAME}:${IMAGE_TAG}" \
  --file "Dockerfile" \
  .)

echo "== 4. Linux App Service plan =="
az appservice plan create \
  --name "$PLAN_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --is-linux \
  --sku "$SKU" \
  --output table

echo "== 5. Web App pointing at the ACR image =="
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)
az webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$PLAN_NAME" \
  --name "$APP_NAME" \
  --deployment-container-image-name "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --output table

echo "== 6. Let the Web App pull from ACR (admin credentials — fine for a course, use a managed identity in real prod) =="
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" --output tsv)
az webapp config container set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --container-image-name "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --container-registry-url "https://${ACR_LOGIN_SERVER}" \
  --container-registry-user "$ACR_USERNAME" \
  --container-registry-password "$ACR_PASSWORD" \
  --output none
echo "Registry auth configured (credentials withheld from output on purpose)."

echo "== 7. App settings: the port Azure should talk to, plus your API keys as secrets =="
# WEBSITES_PORT must match entrypoint.sh's Streamlit fallback port (8080) — NOT
# FastAPI's fixed internal 8000, or Streamlit fails to bind ("port already in use").
# OPENAI_MODEL / OPENAI_EMBEDDING_MODEL are forwarded too: llm.py reads OPENAI_MODEL
# directly and 400s with "you must provide a model parameter" if it's unset.
az webapp config appsettings set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --settings \
    WEBSITES_PORT=8080 \
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
    OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY before running}" \
    COHERE_API_KEY="${COHERE_API_KEY:?set COHERE_API_KEY before running}" \
    OPENAI_MODEL="${OPENAI_MODEL:?set OPENAI_MODEL before running, e.g. gpt-4o-mini}" \
    OPENAI_EMBEDDING_MODEL="${OPENAI_EMBEDDING_MODEL:-}" \
    LANGSMITH_API_KEY="${LANGSMITH_API_KEY:-}" \
    LANGCHAIN_PROJECT="${LANGCHAIN_PROJECT:-multidoc-assist-azure}" \
  --output none
echo "App settings updated (values withheld from output on purpose)."

echo "== 8. Stream logs so you can watch startup (Ctrl+C to stop watching, app keeps running) =="
echo "Site: https://${APP_NAME}.azurewebsites.net"
echo "Run this to tail logs any time:"
echo "  az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo
echo "Done. First load takes 1-2 minutes while the container pulls and both processes boot."
