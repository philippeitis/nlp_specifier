To run cargo flamegraph:

```console
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
cargo flamegraph --bin x -- args
```

Notes:
- phf does not appear to influence the performance of matching terminals to symbols

### Launching NLP Server (For WordNet parsing)
This command will launch StanfordCoreNLP. This is not necessary to use the NLP parser.
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

