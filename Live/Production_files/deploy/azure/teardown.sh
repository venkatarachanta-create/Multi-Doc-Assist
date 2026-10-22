#!/bin/bash
# Deletes everything created by deploy.sh in one shot — run this at the end of
# class (or between runs) so nobody pays for an idle B1 plan overnight.
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-multidoc-assist}"

echo "This will permanently delete resource group: $RESOURCE_GROUP"
read -p "Type the resource group name to confirm: " CONFIRM
if [ "$CONFIRM" != "$RESOURCE_GROUP" ]; then
  echo "Names didn't match — aborting, nothing deleted."
  exit 1
fi

az group delete --name "$RESOURCE_GROUP" --yes --no-wait
echo "Deletion started (--no-wait). Check progress with: az group show --name $RESOURCE_GROUP"
