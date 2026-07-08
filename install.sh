#!/usr/bin/env bash
set -euo pipefail

# --- die must be defined before first use ---
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

(( BASH_VERSINFO[0] >= 4 )) || die "Bash 4+ required."

on_error() {
    printf 'Installation failed on line %d\n' "$1" >&2
}
trap 'on_error $LINENO' ERR

readonly RELEASE=1
readonly REPO_URL="https://github.com/Slupersfe/aaugs-code.git"
readonly RUSTUP_URL="https://sh.rustup.rs"

readonly VIBE_DIR="${HOME}/vibe"
readonly INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/aaugs-code"

# --- Flags ---
VERBOSE=false
FORCE=false
UNINSTALL=false
PREFIX=""

# --- Help ---
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --prefix DIR    Install binary into DIR (auto-detected by default)
  --force         Overwrite existing configs and allow dirty repo
  --uninstall     Remove installed binary and data
  --verbose       Print detailed progress
  --help          Show this help
EOF
    exit 0
}

# --- Logging ---
info() { printf '==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
error() { printf 'ERROR: %s\n' "$*" >&2; }
debug() { if "$VERBOSE"; then printf 'DEBUG: %s\n' "$*"; fi; }

# --- Require helper ---
require() {
    command -v "$1" >/dev/null || die "$1 is required but not found."
}

# --- Parse CLI ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || die "--prefix requires a directory argument."
            PREFIX="$2"
            shift 2
            ;;
        --force) FORCE=true; shift ;;
        --uninstall) UNINSTALL=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help) usage ;;
        *) die "Unknown option: $1. Use --help for usage." ;;
    esac
done

# --- Sudo handling ---
if (( EUID == 0 )); then
    SUDO=""
else
    require sudo
    SUDO="sudo"
fi

# --- Validate --prefix ---
if [ -n "$PREFIX" ]; then
    mkdir -p "$PREFIX"
    [[ -d "$PREFIX" ]] || die "Cannot create or access --prefix directory: $PREFIX"
fi

# --- Cleanup ---
tmp_rustup=""
cleanup() {
    if [ -n "$tmp_rustup" ] && [ -f "$tmp_rustup" ]; then
        rm -f "$tmp_rustup"
    fi
}
trap cleanup EXIT

# --- Uninstall ---
if "$UNINSTALL"; then
    info "Uninstalling..."

    candidates=(
        "${CARGO_HOME:-$HOME/.cargo}/bin/aaugs-code"
        "$HOME/.local/bin/aaugs-code"
    )

    if [ -n "$PREFIX" ]; then
        candidates+=("${PREFIX}/bin/aaugs-code")
    fi

    for bin in "${candidates[@]}"; do
        if [ -f "$bin" ]; then
            rm -f "$bin"
            info "Removed $bin"
        fi
    done

    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
        info "Removed $INSTALL_DIR"
    fi

    info "Uninstall complete. Config in ${VIBE_DIR}/config was kept."
    exit 0
fi

# --- Prerequisites ---

require curl
require mkdir
require install

# git
if ! command -v git &>/dev/null; then
    info "git not found, installing..."
    case "$(uname -s)" in
        Darwin)
            if command -v brew &>/dev/null; then
                brew install git
            elif command -v xcode-select &>/dev/null; then
                xcode-select --install
            elif command -v port &>/dev/null; then
                $SUDO port install git
            else
                die "no package manager found (brew/xcode-select/port), install git manually"
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                $SUDO apt-get update
                $SUDO apt-get install -y git
            elif command -v dnf &>/dev/null; then
                $SUDO dnf install -y git
            elif command -v yum &>/dev/null; then
                $SUDO yum install -y git
            elif command -v pacman &>/dev/null; then
                $SUDO pacman -Sy --noconfirm git
            elif command -v zypper &>/dev/null; then
                $SUDO zypper install -y git
            else
                die "unsupported Linux package manager, install git manually"
            fi
            ;;
        *)
            die "unsupported OS: $(uname -s), install git manually"
            ;;
    esac
fi
require git

# rust/cargo
if ! command -v cargo &>/dev/null; then
    info "cargo not found, installing rustup..."
    tmp_rustup=$(mktemp)
    curl -fsSL "$RUSTUP_URL" -o "$tmp_rustup"
    sh "$tmp_rustup" -y
    . "${HOME}/.cargo/env"
fi

require cargo
require rustc

cargo --version >/dev/null || die "Cargo installation failed."
rustc --version >/dev/null || die "Rust installation failed."

# --- Network check ---
info "Checking network connectivity..."
git ls-remote "$REPO_URL" >/dev/null || die "Cannot reach $REPO_URL"

# --- Clone / Pull ---

if [ -d "${INSTALL_DIR}/.git" ]; then
    info "Updating existing clone at ${INSTALL_DIR}..."
    origin=$(git -C "$INSTALL_DIR" remote get-url origin)
    if [[ "$origin" != "$REPO_URL" ]]; then
        die "Repository origin '$origin' does not match expected '$REPO_URL'"
    fi

    if [[ -n "$(git -C "$INSTALL_DIR" status --porcelain)" ]]; then
        if "$FORCE"; then
            warn "Local modifications found, discarding (--force)"
        else
            die "Local modifications detected. Use --force to discard."
        fi
    fi

    git -C "$INSTALL_DIR" fetch origin
    git -C "$INSTALL_DIR" remote set-head origin --auto 2>/dev/null || true

    default_branch=$(
        git -C "$INSTALL_DIR" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null \
            | sed 's@^refs/remotes/origin/@@'
    )

    if [[ -z "$default_branch" ]]; then
        default_branch=$(
            git -C "$INSTALL_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null \
            || echo "main"
        )
    fi

    git -C "$INSTALL_DIR" reset --hard "origin/$default_branch"
else
    info "Cloning into ${INSTALL_DIR}..."
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    git clone --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"

# --- Build and install via cargo install ---

info "Building and installing..."

cargo_args=(
    --locked
    --path .
)

if [ -n "$PREFIX" ]; then
    cargo_args+=(--root "$PREFIX")
fi

if "$FORCE"; then
    cargo_args+=(--force)
fi

if ! "$VERBOSE"; then
    cargo_args+=(--quiet)
fi

cargo install "${cargo_args[@]}"

# Determine install location for output
if [ -n "$PREFIX" ]; then
    install_bin="${PREFIX}/bin"
else
    install_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
fi

installed_path="${install_bin}/aaugs-code"
[[ -x "$installed_path" ]] || die "Installation completed but binary is missing."
echo "Installed: ${installed_path}"

if [[ ":$PATH:" != *":${install_bin}:"* ]]; then
    warn "${install_bin} is not on your PATH. Add it to your shell rc file."
fi

# --- Config dirs ---

info "Setting up config directories..."
mkdir -p "${VIBE_DIR}/config"
mkdir -p "${VIBE_DIR}/sessions"

# --- Release config (idempotent unless --force) ---

readonly release_file="${VIBE_DIR}/release.config"
if [[ ! -f "$release_file" || "$FORCE" == true ]]; then
    info "Writing release.config..."
    printf '%s\n' "$RELEASE" >"$release_file"
else
    debug "release.config already exists at ${release_file}, skipping"
fi

# --- Done ---

echo ""
echo "Installation complete!"
echo "  Binary: aaugs-code (${installed_path})"
echo "  Config: ${VIBE_DIR}/config/"
echo "  Sessions: ${VIBE_DIR}/sessions/"
echo "  Release: ${release_file}"
echo ""
echo "Run 'aaugs-code --help' to get started."
