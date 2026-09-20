#!/bin/bash
# E9: per-worker (thread_local) cache sizing at W=64.
#
# Each worker holds ~27 MB of private caches; 8 workers share one CCDs 32 MiB
