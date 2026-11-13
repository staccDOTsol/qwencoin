#!/bin/bash
# Run on the Mac that holds the Developer ID. It puts what the build workflow needs to sign (and to
# notarize) the macOS programs into the repository's Actions secrets, from this Mac straight to
# GitHub: `gh` encrypts each value for the repository before it leaves. No value is printed, and
# what is written to disk on the way is deleted at the end.
#
#   bash tools/mac-signing-secrets.sh [OWNER/REPO]
#   P12=/path/to/exported.p12 [P12_PASSWORD=...] bash tools/mac-signing-secrets.sh    an identity already exported
#   KEYCHAIN=1 bash tools/mac-signing-secrets.sh                                      export it from the keychain now
#
# Without P12, *.p12 files in ~/.developer-id are looked at. A file's password is taken from P12_PASSWORD,
# else looked for in the small files next to it, else asked for. If the file cannot be opened, or there is
# none, the identity is exported from the login keychain: macOS then asks for the login password, once per
# signing key the keychain holds. Whatever comes out, only the "Developer ID Application" certificate and
# its own key go to GitHub.
#
# Notarization, if one of these is found (else the programs are signed and not notarized):
#   an App Store Connect API key: API_KEY=/path/AuthKey_KEYID.p8 (or the one in ~/.private_keys or
#   ~/.appstoreconnect/private_keys) with API_ISSUER=its-issuer-uuid
#   or APPLE_ID=... APPLE_TEAM_ID=... APPLE_APP_PASSWORD=an-app-specific-password
#
# Secrets written: MACOS_CERT_P12_BASE64, MACOS_CERT_PASSWORD, and for notarization APPLE_API_KEY_P8,
# APPLE_API_KEY_ID, APPLE_API_ISSUER_ID (or APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD).
set -euo pipefail
REPO="${1:-staccDOTsol/qwencoin}"
say() { echo "signing-secrets: $*"; }
die() { echo "signing-secrets: $*" >&2; exit 1; }
ask() {  # ask PROMPT: one line from the keyboard, not echoed (this script may itself be arriving on stdin)
  local x=""
  { : < /dev/tty; } 2>/dev/null || return 1
  printf '%s' "$1" > /dev/tty; IFS= read -r -s x < /dev/tty || true; echo > /dev/tty
  printf '%s' "$x"
}
command -v gh >/dev/null 2>&1 || die "gh (GitHub's command line) is needed here:  brew install gh && gh auth login"
gh auth status >/dev/null 2>&1 || die "gh is not signed in:  gh auth login"
gh api "repos/$REPO" --jq .full_name >/dev/null 2>&1 || die "gh cannot see $REPO with this login"
SSL=/usr/bin/openssl; [ -x "$SSL" ] || SSL=openssl
T="$(mktemp -d)"; chmod 700 "$T"; trap 'rm -rf "$T"' EXIT
opens() { "$SSL" pkcs12 -in "$1" -passin "pass:$2" -nodes -out "$T/all.pem" 2>/dev/null || "$SSL" pkcs12 -legacy -in "$1" -passin "pass:$2" -nodes -out "$T/all.pem" 2>/dev/null; }

# ---- the identity: a file already exported, or out of the keychain
P12="${P12:-}"; PASS="${P12_PASSWORD:-}"
if [ -z "$P12" ] && [ -z "${KEYCHAIN:-}" ] && [ -d "$HOME/.developer-id" ]; then
  FOUND=(); while IFS= read -r f; do FOUND+=("$f"); done < <(find "$HOME/.developer-id" -maxdepth 3 -type f -name '*.p12' 2>/dev/null | sort)
  if [ "${#FOUND[@]}" = 1 ]; then P12="${FOUND[0]}"; say "an exported identity is here: $P12"
  elif [ "${#FOUND[@]}" -gt 1 ]; then say "several exported files in ~/.developer-id:"; printf '   %s\n' "${FOUND[@]}"; die "name one:  P12=... bash $0"; fi
fi
OPEN=""
if [ -n "$P12" ]; then
  [ -f "$P12" ] || die "$P12 is not a file"
  if opens "$P12" "$PASS"; then OPEN=1
  else  # a password kept in a small file next to it: every line, and what stands after NAME= or NAME: on a line, is tried
    PASS=""
    while IFS= read -r f; do
      while IFS= read -r c; do
        if [ -n "$c" ] && opens "$P12" "$c"; then PASS="$c"; OPEN=1; say "its password is in $f"; break 2; fi
      done < <(tr -d '\r' < "$f" | sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' | awk 'NF {print; x=$0; if (sub(/^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*[=:][[:space:]]*/, "", x)) {print x; gsub(/^["\047]|["\047],?$/, "", x); print x}}')
    done < <(find "$(dirname "$P12")" -maxdepth 1 -type f -size -8k ! -name '*.p12' ! -name '*.cer' ! -name '*.pem' ! -name '*.p8' ! -name '*.key' ! -name '.DS_Store' 2>/dev/null)
    if [ -z "$OPEN" ]; then
      PASS="$(ask "password of $(basename "$P12") (not shown; just Enter if you do not have it): ")" || PASS=""
      if [ -n "$PASS" ] && opens "$P12" "$PASS"; then OPEN=1; else say "$(basename "$P12") does not open: the identity is taken from the keychain instead"; fi
    fi
  fi
fi
if [ -z "$OPEN" ]; then
  command -v security >/dev/null 2>&1 || die "this is not a Mac (no security tool), and there is no exported identity that opens"
  ID="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
  [ -n "$ID" ] || die "this keychain has no 'Developer ID Application' identity (security find-identity -v -p codesigning)"
  N="$(security find-identity -v -p codesigning | grep -c '^ *[0-9]*)' || true)"
  say "exporting from the keychain: $ID"
  say "macOS now asks for your login password to allow it: up to $N times, once per signing key in the keychain (choose Allow each time; only the Developer ID one leaves this Mac)"
  PASS="$("$SSL" rand -hex 24)"
  security export -k "$(security default-keychain | sed 's/^ *"//; s/" *$//')" -t identities -f pkcs12 -P "$PASS" -o "$T/keychain.p12" \
    || die "the export did not go through. By hand: Keychain Access, My Certificates, right-click the Developer ID Application entry, Export (.p12, with a password), then  P12=that-file bash $0"
  opens "$T/keychain.p12" "$PASS" || die "the exported file could not be read back"
fi

# ---- of what the file holds, only the Developer ID Application certificate and the key that belongs to it
awk -v d="$T" '/-----BEGIN /{n++; f=sprintf("%s/part%03d.pem", d, n)} f{print > f} /-----END /{close(f); f=""}' "$T/all.pem"
CERT=""; KEY=""; : > "$T/chain.pem"
for f in "$T"/part*.pem; do
  if grep -q "BEGIN CERTIFICATE" "$f"; then
    S="$("$SSL" x509 -in "$f" -noout -subject 2>/dev/null || true)"
    case "$S" in
      *"Developer ID Application"*) [ -n "$CERT" ] || CERT="$f" ;;
      *"Developer ID Certification Authority"*) cat "$f" >> "$T/chain.pem" ;;
    esac
  fi
done
[ -n "$CERT" ] || die "there is no 'Developer ID Application' certificate in it (it holds: $(for f in "$T"/part*.pem; do grep -q "BEGIN CERTIFICATE" "$f" && "$SSL" x509 -in "$f" -noout -subject 2>/dev/null | sed 's/.*CN *= *//; s/[,/].*//'; done | tr '\n' ';'))"
WANT="$("$SSL" x509 -in "$CERT" -noout -pubkey | "$SSL" sha256)"
for f in "$T"/part*.pem; do
  if grep -q "PRIVATE KEY" "$f" && [ "$("$SSL" pkey -in "$f" -pubout 2>/dev/null | "$SSL" sha256)" = "$WANT" ]; then KEY="$f"; fi
done
[ -n "$KEY" ] || die "the file has the Developer ID Application certificate without its private key: export it together with its key"
NAME="$("$SSL" x509 -in "$CERT" -noout -subject | sed 's/.*CN *= *//; s/, *[A-Z]* *=.*//; s#/[A-Z]*=.*##' | tr -d '"')"
END="$("$SSL" x509 -in "$CERT" -noout -enddate | cut -d= -f2)"
"$SSL" x509 -in "$CERT" -noout -checkend 0 >/dev/null || die "the certificate '$NAME' expired on $END"
say "the identity: $NAME (valid until $END)"
NEW="$("$SSL" rand -hex 24)"
if [ -s "$T/chain.pem" ]; then "$SSL" pkcs12 -export -inkey "$KEY" -in "$CERT" -certfile "$T/chain.pem" -name "$NAME" -out "$T/id.p12" -passout "pass:$NEW"
else "$SSL" pkcs12 -export -inkey "$KEY" -in "$CERT" -name "$NAME" -out "$T/id.p12" -passout "pass:$NEW"; fi
base64 < "$T/id.p12" | tr -d '\n' | gh secret set MACOS_CERT_P12_BASE64 --repo "$REPO"
printf '%s' "$NEW" | gh secret set MACOS_CERT_PASSWORD --repo "$REPO"
say "the identity is in $REPO's secrets (under a password made here and stored beside it)"

# ---- notarization
API_KEY="${API_KEY:-}"
if [ -z "$API_KEY" ]; then
  for d in "$HOME/.private_keys" "$HOME/private_keys" "$HOME/.appstoreconnect/private_keys"; do
    for f in "$d"/AuthKey_*.p8; do [ -f "$f" ] && [ -z "$API_KEY" ] && API_KEY="$f"; done
  done
fi
if [ -n "$API_KEY" ] && [ -f "$API_KEY" ]; then
  KID="$(basename "$API_KEY" .p8)"; KID="${KID#AuthKey_}"
  ISS="${API_ISSUER:-}"
  [ -n "$ISS" ] || ISS="$(ask "issuer id of the App Store Connect key $KID (the UUID above the key list in App Store Connect; empty: no notarization): ")" || ISS=""
  if [ -n "$ISS" ]; then
    gh secret set APPLE_API_KEY_P8 --repo "$REPO" < "$API_KEY"
    printf '%s' "$KID" | gh secret set APPLE_API_KEY_ID --repo "$REPO"
    printf '%s' "$ISS" | gh secret set APPLE_API_ISSUER_ID --repo "$REPO"
    say "notarization: the App Store Connect key $KID is in the secrets"
  else say "notarization: no issuer id given, so not set up (API_ISSUER=... bash $0 adds it)"; fi
elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
  printf '%s' "$APPLE_ID" | gh secret set APPLE_ID --repo "$REPO"
  printf '%s' "$APPLE_TEAM_ID" | gh secret set APPLE_TEAM_ID --repo "$REPO"
  printf '%s' "$APPLE_APP_PASSWORD" | gh secret set APPLE_APP_PASSWORD --repo "$REPO"
  say "notarization: the Apple ID and its app-specific password are in the secrets"
else
  say "notarization: nothing found for it (an AuthKey_*.p8 with API_ISSUER=..., or APPLE_ID, APPLE_TEAM_ID and APPLE_APP_PASSWORD): the programs will be signed, not notarized"
fi
say "the repository's secrets now (names only):"
gh secret list --repo "$REPO" | awk '{print "   " $1}'
if gh workflow run node.yml --repo "$REPO" >/dev/null 2>&1; then say "the build workflow is started again: https://github.com/$REPO/actions"
else say "start the build:  gh workflow run node.yml --repo $REPO"; fi
