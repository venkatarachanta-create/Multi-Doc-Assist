#!/bin/bash
# Loads a python-dotenv-style .env file into the current shell's environment.
#
# Plain `source .env` breaks on lines like `KEY = "value"` (spaces around `=`,
# quoted values) — valid for python-dotenv/pydantic-settings, not for bash.
# This parses that format properly instead.
#
# Usage: source load_env.sh ../../src/backend/.env
set -a
while IFS= read -r line || [ -n "$line" ]; do
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac

    key="${line%%=*}"
    value="${line#*=}"
    key="$(echo "$key" | sed -e 's/[[:space:]]*$//')"
    value="$(echo "$value" | sed -e 's/^[[:space:]]*//')"
    # strip one layer of surrounding quotes, if present
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"

    export "$key=$value"
done < "$1"
set +a
