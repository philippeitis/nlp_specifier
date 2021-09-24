# Installation
Python 3.6+ and Rust should already be installed for the parsing steps. To perform verification of output specifcations, Prusti should also be installed.
Dependencies for individual components of the system are specified below, or are otherwise manually installed using provided `setup.sh` scripts.
Note that all .sh files and commands provided are specific to Linux. 

## Python Implementation
The python implement is available at https://github.com/philippeitis/nlp_specifier/tree/b42778e2cb51e5d8edf08c0cc7a5060225468d92.
## HTML Documentation Mining
This project provides functionality for extracting Rust documentation from pages output by cargo docs, and for documentation downloaded via `rustup`.
To set up parsing of HTML documentation, build [src/doc_parser/](src/doc_parser/):
```bash
cd ./src/doc_parser/ && cargo build --release ; cd ..
```

## NLP Parser
The NLP parsing code will tokenize, assign parts of speech tags, and generate parse-trees for selected functions in a particular file of Rust source code.
To set up the NLP parser, use [doc_parser/setup.sh](doc_parser/setup.sh):
```bash
cd ./nlp/ && sudo chmod +x ./setup.sh && ./setup.sh && cd .
```

Optionally, it may be useful to review these links:

https://www.nltk.org/install.html

https://spacy.io/usage

### WordNet POS tags reference
This link provides a useful reference for the POS tags generated by spaCy:
http://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm

[comment]: <> (### Launching NLP Server &#40;For WordNet parsing&#41;)

[comment]: <> (This command will launch StanfordCoreNLP. This is not necessary to use the NLP parser.)

[comment]: <> (```bash)

[comment]: <> (java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000)

[comment]: <> (```)


## Rust AST Parser
The Rust AST parser transforms input Rust files into an AST, which is used in the NLP parser to generate specifications.
To set up the Rust AST parser, build [src/doc_parser/](src/doc_parser/):
```bash
cd ./src/doc_parser/ && cargo build --release ; cd ..
```

## Named-entity Recognition and Semantic Role Labelling
### Requirements
To use NER and SRL analysis for documentation, Docker and Docker Compose must be installed. Additionally, downloading the relevant models requires installing Git,
and Git LFS. All other dependencies for this are set up using [jml_nlp/setup.sh](jml_nlp/setup.sh).
```bash
cd ./jml_nlp/ && sudo chmod +x ./setup.sh && ./setup.sh && cd .
```
After running this script, the SRL service will be available at 127.0.0.8:701, and the NER service will be available at 127.0.0.8:702.
[src/nlp/ner.py](src/nlp/ner.py) provides functions for annotating text using these services. The Tokenizer class in [src/nlp/tokenizer.py](doc_parser/doc_parser.py) transforms these annotations to a format that can be rendered by spaCy's displaCy tool.

The NER and SRL models are sourced from `Combining formal and machine learning techniques for the generation of JML specifications`.

# Usage
Once installation is complete, this project can be used through `doc_parser`. Run the following command to see a list of all possible commands.
```console
foo@bar:~$ python -m doc_parser --help
Usage: __main__.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  end-to-end  Demonstrates entire pipeline from end to end on provided file.
  render      Visualization of various components in the system's pipeline.
  specify     Creates specifications for a variety of sources.
```

To see more specific help, do the following:
```console
foo@bar:~$ python -m doc_parser end-to-end --help
Usage: __main__.py end-to-end [OPTIONS]

  Demonstrates entire pipeline from end to end on provided file.

Options:
  -p, --path PATH  Source file to specify.
  --help           Show this message and exit.
```
