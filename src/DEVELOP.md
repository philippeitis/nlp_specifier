To run cargo flamegraph:

```console
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
cargo flamegraph --bin x -- args
```

Notes:
- phf does not appear to influence the performance of matching terminals to symbols