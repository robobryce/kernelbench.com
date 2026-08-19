#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage:
  scripts/reward_hacking_report.sh render REPORT.md
  scripts/reward_hacking_report.sh verify REPORT.md [SHA256SUMS]
EOF
    exit 2
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

resolve_report() {
    local supplied=$1
    if [[ $supplied != /* ]]; then
        supplied="$PWD/$supplied"
    fi
    [[ -f $supplied ]] || die "report does not exist: $1"

    REPORT_DIR="$(CDPATH='' cd -- "$(dirname -- "$supplied")" && pwd)"
    REPORT_NAME="$(basename -- "$supplied")"
    case $REPORT_NAME in
        *-report-reward-hacking.md)
            OUTPUT=${REPORT_NAME%-report-reward-hacking.md}
            ;;
        *)
            die "report name must end with -report-reward-hacking.md"
            ;;
    esac
    [[ -n $OUTPUT ]] || die "report output name cannot be empty"

    REPORT="$REPORT_DIR/$REPORT_NAME"
    HTML="$REPORT_DIR/$OUTPUT-report-reward-hacking.html"
}

validate_schema() {
    autocuda schema check report-reward-hacking-markdown \
        --data-dir "$REPORT_DIR" \
        --output "$OUTPUT"
}

verify_checksums() {
    local supplied=$1
    if [[ $supplied != /* ]]; then
        supplied="$PWD/$supplied"
    fi
    [[ -f $supplied ]] || die "checksum file does not exist: $1"

    local checksum_dir checksum_name
    checksum_dir="$(CDPATH='' cd -- "$(dirname -- "$supplied")" && pwd)"
    checksum_name="$(basename -- "$supplied")"

    local -a checker
    if command -v sha256sum >/dev/null 2>&1; then
        checker=(sha256sum --check)
    elif command -v shasum >/dev/null 2>&1; then
        checker=(shasum --algorithm 256 --check)
    else
        die "sha256sum or shasum is required"
    fi

    (cd "$checksum_dir" && "${checker[@]}" "$checksum_name")
}

[[ $# -ge 2 ]] || usage
COMMAND=$1
resolve_report "$2"
require_command autocuda

case $COMMAND in
    render)
        [[ $# -eq 2 ]] || usage
        require_command pandoc
        validate_schema
        autocuda report html "$REPORT"
        [[ -s $HTML ]] || die "renderer did not create HTML: $HTML"
        ;;
    verify)
        [[ $# -le 3 ]] || usage
        validate_schema
        [[ -s $HTML ]] || die "generated HTML does not exist: $HTML"
        verify_checksums "${3:-$REPORT_DIR/SHA256SUMS}"
        ;;
    *)
        usage
        ;;
esac
