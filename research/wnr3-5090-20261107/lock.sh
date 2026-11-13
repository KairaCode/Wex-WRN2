# heavy.lock helper (same protocol as lab/v41-gpu-20261107/w1/take_lock.ps1): atomic create (O_EXCL via noclobber),
# poll every 60 s up to 90 min, remove on exit. Source it, then: take_lock "label" || exit 3
LOCK='<KAIRA>'/lab/v41-gpu-20261107/heavy.lock
take_lock() {
  local deadline=$((SECONDS + 5400))
  while ! ( set -o noclobber; printf '%s %s\n' "$1" "$(date +%FT%T)" > "$LOCK" ) 2>/dev/null; do
    [ $SECONDS -gt $deadline ] && { echo "lock: gave up after 90 min (held by: $(cat "$LOCK" 2>/dev/null))"; return 1; }
    sleep 60
  done
  echo "lock: taken $(date +%T)"; trap 'rm -f "$LOCK"; echo "lock: released $(date +%T)"' EXIT
}
