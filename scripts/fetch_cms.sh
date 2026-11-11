#!/usr/bin/env bash
# Download the public CMS inputs (~60 MB). Only the reference tables and test claims inside the
# grouper package are extracted; the bundled Java software is never run.
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p data/raw/cms && cd data/raw/cms
UA="Mozilla/5.0 (period-audit research script)"
curl -sSL -A "$UA" -o grouper_v07226.zip "https://www.cms.gov/files/zip/oct-2026-hh-pps-grouper-software-hh-pdgm-v07-2-26-posted-08-20-2026.zip"
curl -sSL -A "$UA" -o pdgm_cy2026.zip "https://www.cms.gov/files/zip/cy-2026-final-hh-pdgm-case-mix-weights-lupa-thresholds.zip"
curl -sSL -A "$UA" -o comorbidity_cy2026.zip "https://www.cms.gov/files/zip/cy-2026-final-low-comorbidity-high-comorbidity-adjustment-interactions.zip"
unzip -oq grouper_v07226.zip "HomeHealthGrouperSoftware/tables/*" "HomeHealthGrouperSoftware/test/*" "HomeHealthGrouperSoftware/documentation/*" -d grouper
unzip -oq pdgm_cy2026.zip -d pdgm
unzip -oq comorbidity_cy2026.zip -d comorb
echo "CMS files in data/raw/cms/"
