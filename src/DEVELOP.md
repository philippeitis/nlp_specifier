## Launching Server
### Python
```console
python ./nlp/server.py -p 5000 -h 0.0.0.0
```

### Docker
```console
cd ./nlp
sudo docker-compose up --build
```

## Formatting
### Python
```console
pip install -U black isort
black target && isort target
```

### Rust
```console
cargo fix && cargo fmt
```

## Optimization Notes
- Using msgpack to minimize message size and overhead
- phf (Rust) has no measurable impact on turning strings into terminals

### Rust
```console
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
cargo flamegraph --bin x -- args
```
