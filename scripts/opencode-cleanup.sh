#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
ts() { date +"%Y%m%d-%H%M%S"; }

CONFIG=""
RESTORE=""
VALIDATE_URLS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)   CONFIG="${2:-}"; shift 2;;
    --restore)  RESTORE="${2:-}"; shift 2;;
    --validate) VALIDATE_URLS+=("${2:-}"); shift 2;;
    -h|--help)
      cat <<'EOF'
Usage:
  opencode-cleanup.sh --config <path/to/opencode.jsonc> [--validate <baseURL>]...
  opencode-cleanup.sh --restore <backup_dir>
EOF
      exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

copy_if_exists() { [[ -e "$1" ]] && mkdir -p "$2" && cp -a "$1" "$2/"; }
move_if_exists() { [[ -e "$1" ]] && mv "$1" "${1}.disabled.${2}" && echo "Moved: $1 -> ${1}.disabled.${2}"; }

python_edit_config() {
  local file="$1"
  [[ -f "$file" ]] || die "Config file not found: $file"

  python3 - "$file" <<'PY'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
txt = path.read_text(encoding="utf-8")

def ensure_kv(txt: str, key: str, value: str) -> str:
  # Replace first occurrence of "key": ... (best-effort JSONC text edit)
  pat = re.compile(r'(^\s*"' + re.escape(key) + r'"\s*:\s*)(.*?)(\s*,\s*)?$', re.M | re.S)
  m = pat.search(txt)
  if m:
    indent = re.match(r'^\s*', m.group(0)).group(0)
    suffix = "," if (m.group(3) is not None) else ""
    replacement = f'{indent}"{key}": {value}{suffix}'
    return txt[:m.start()] + replacement + txt[m.end():]

  brace = txt.find("{")
  if brace == -1:
    raise SystemExit("Config does not contain '{'")
  insert = f'\n  "{key}": {value},'
  return txt[:brace+1] + insert + txt[brace+1:]

txt = ensure_kv(txt, "plugin", "[]")
txt = ensure_kv(txt, "autoupdate", "false")
path.write_text(txt, encoding="utf-8")
print(f"Patched: {path}")
PY
}

validate_models() {
  local base="$1"
  local url="${base%/}/models"
  echo "==> GET $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" | head -c 2000 || true; echo
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$url" | head -c 2000 || true; echo
  fi
}

restore_backup() {
  local dir="$1"
  [[ -d "$dir" ]] || die "Backup dir not found: $dir"
  echo "Restoring from: $dir"

  [[ -d "$dir/repo/.opencode" ]] && rm -rf .opencode && cp -a "$dir/repo/.opencode" .opencode
  [[ -n "${CONFIG:-}" && -f "$dir/repo/config.backup" ]] && cp -a "$dir/repo/config.backup" "$CONFIG"

  [[ -f "$dir/global/opencode.jsonc" ]] && mkdir -p ~/.config/opencode && cp -a "$dir/global/opencode.jsonc" ~/.config/opencode/opencode.jsonc
  [[ -d "$dir/global/plugins" ]] && mkdir -p ~/.config/opencode && rm -rf ~/.config/opencode/plugins && cp -a "$dir/global/plugins" ~/.config/opencode/plugins

  echo "Restore complete."
}

if [[ -n "$RESTORE" ]]; then
  restore_backup "$RESTORE"
  exit 0
fi

[[ -n "$CONFIG" ]] || die "Missing --config <path>"

STAMP="$(ts)"
BACKUP=".opencode-backup/${STAMP}"
mkdir -p "$BACKUP/repo" "$BACKUP/global"
echo "Backup directory: $BACKUP"

# backups
copy_if_exists "$CONFIG" "$BACKUP/repo"
[[ -f "$BACKUP/repo/$(basename "$CONFIG")" ]] && mv "$BACKUP/repo/$(basename "$CONFIG")" "$BACKUP/repo/config.backup"
copy_if_exists ".opencode" "$BACKUP/repo" || true

copy_if_exists "$HOME/.config/opencode/opencode.jsonc" "$BACKUP/global" || true
[[ -d "$HOME/.config/opencode/plugins" ]] && cp -a "$HOME/.config/opencode/plugins" "$BACKUP/global/plugins" || true
[[ -d "$HOME/.config/opencode/node_modules" ]] && cp -a "$HOME/.config/opencode/node_modules" "$BACKUP/global/node_modules" || true
[[ -f "$HOME/.config/opencode/package.json" ]] && cp -a "$HOME/.config/opencode/package.json" "$BACKUP/global/" || true
[[ -f "$HOME/.config/opencode/bun.lock" ]] && cp -a "$HOME/.config/opencode/bun.lock" "$BACKUP/global/" || true

# disable plugins folders (old path)
move_if_exists ".opencode/plugins" "$STAMP"
move_if_exists "$HOME/.config/opencode/plugins" "$STAMP"

# disable “workspace” plugin installs (the real culprit in your case)
move_if_exists ".opencode/node_modules" "$STAMP"
move_if_exists ".opencode/package.json" "$STAMP"
move_if_exists ".opencode/bun.lock" "$STAMP"

move_if_exists "$HOME/.config/opencode/node_modules" "$STAMP"
move_if_exists "$HOME/.config/opencode/package.json" "$STAMP"
move_if_exists "$HOME/.config/opencode/bun.lock" "$STAMP"

# patch config
python_edit_config "$CONFIG"

# validate endpoints if requested
for u in "${VALIDATE_URLS[@]}"; do
  validate_models "$u"
done

cat <<EOF

Done.

Try:
  OPENCODE_CONFIG=$CONFIG opencode

Restore:
  ./scripts/opencode-cleanup.sh --config $CONFIG --restore $BACKUP

EOF
