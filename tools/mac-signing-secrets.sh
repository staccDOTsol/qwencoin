#!/bin/bash
# Run on the Mac that holds the Developer ID. It puts what the build workflow needs to sign (and to
# notarize) the macOS programs into the repository's Actions secrets, from this Mac straight to
# GitHub: `gh` encrypts each value for the repository before it leaves. No value is printed, and
# what is written to disk on the way is deleted at the end.
#
#   bash tools/mac-signing-secrets.sh [OWNER/REPO]
#   P12=/path/to/exported.p12 [P12_PASSWORD=...] bash tools/mac-signing-secrets.sh    an identity already exported
#
# Without P12, *.p12 files in ~/.developer-id are looked at; with none there, the identity is exported
# from the login keychain (macOS asks to allow it). Whatever the file holds, only the "Developer ID
# Application" certificate and its own key go to GitHub.
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
  [ -r /dev/tty ] || return 1
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
if [ -z "$P12" ] && [ -d "$HOME/.developer-id" ]; then
  FOUND=(); while IFS= read -r f; do FOUND+=("$f"); done < <(find "$HOME/.developer-id" -maxdepth 3 -type f -name '*.p12' 2>/dev/null | sort)
  if [ "${#FOUND[@]}" = 1 ]; then P12="${FOUND[0]}"; say "an exported identity is here: $P12"
  elif [ "${#FOUND[@]}" -gt 1 ]; then say "several exported files in ~/.developer-id:"; printf '   %s\n' "${FOUND[@]}"; die "name one:  P12=... bash $0"; fi
fi
if [ -n "$P12" ]; then
  [ -f "$P12" ] || die "$P12 is not a file"
  if [ -z "$PASS" ] && ! opens "$P12" ""; then  # a password kept next to it, else asked for
    for f in "${P12%.p12}.password" "${P12%.p12}.pass" "${P12%.p12}.txt" "$(dirname "$P12")/password" "$(dirname "$P12")/password.txt" "$(dirname "$P12")/.password"; do
      [ -f "$f" ] && PASS="$(head -1 "$f" | tr -d '\r\n')" && opens "$P12" "$PASS" && { say "its password is in $f"; break; }
      PASS=""
    done
    [ -n "$PASS" ] || PASS="$(ask "password of $(basename "$P12") (not shown): ")" || die "the password is needed:  P12_PASSWORD=... bash $0"
  fi
  opens "$P12" "$PASS" || die "$P12 does not open with that password"
else
  ID="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
  [ -n "$ID" ] || die "this keychain has no 'Developer ID Application' identity (security find-identity -v -p codesigning), and no P12=file was named"
  say "exporting from the keychain: $ID  (macOS asks to allow it, once per key it holds)"
  PASS="$("$SSL" rand -hex 24)"
  security export -k "$(security default-keychain | sed 's/^ *"//; s/" *$//')" -t identities -f pkcs12 -P "$PASS" -o "$T/keychain.p12" || die "the export was not allowed"
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
