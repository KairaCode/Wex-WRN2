#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The machine lock of the self-hosted CI runner: a job on that runner takes it before its first check and gives it
# back in a last step that runs on failure and cancel too, so a CI job never runs while the machine's owner runs
# something else there. The lock is a directory (mkdir is atomic); its path comes from CI_MACHINE_LOCK, which the
# runner sets (the .env file of the runner), never from the repository. Not used on GitHub-hosted runners.
#
#   bash scripts/ci_lock.sh take       wait until the directory is absent, create it and write owner.txt
#                                      ("github-ci <run id>-<attempt>"); the calling step's timeout-minutes
#                                      bounds the wait
#   bash scripts/ci_lock.sh release    remove it, only if owner.txt still names this run
set -u
lock=${CI_MACHINE_LOCK:?CI_MACHINE_LOCK is not set on this runner: no lock, no job}
me="github-ci ${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
case ${1:-} in
  take)
    waited=0
    until mkdir "$lock" 2> /dev/null; do
      if [ $((waited % 300)) = 0 ]; then
        echo "lock held by: $(cat "$lock/owner.txt" 2> /dev/null || echo '(no owner.txt)'); waiting"
      fi
      sleep 15; waited=$((waited + 15))
    done
    echo "$me" > "$lock/owner.txt"
    echo "lock taken after ${waited} s: $me" ;;
  release)
    if [ "$(cat "$lock/owner.txt" 2> /dev/null)" = "$me" ]; then
      rm -rf -- "$lock" && echo "lock released"
    else
      echo "lock not ours (owner: $(cat "$lock/owner.txt" 2> /dev/null || echo none)): left as it is"
    fi ;;
  *) echo "usage: $0 take|release" >&2; exit 2 ;;
esac
