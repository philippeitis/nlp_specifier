### Launching NLP Server (For WordNet parsing)

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### WordNet POS tags reference
http://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm

### Flair POS tags reference
https://huggingface.co/flair/pos-english

### Flair Github Link

### NLTK Tree parsing
https://stackoverflow.com/questions/24975573/how-to-parse-custom-tags-using-nltk-regexp-parser
https://www.nltk.org/book/ch08.html

### Set up
Python 3.6+ and Rust should already be setup for the parsing steps. For the verification step,
Prusti should also be installed.

## NLP Parser
To set up the NLP parser, the following Python dependencies will need to be installed.
```bash
pip install flair
pip install nltk
```

Further instructions are available at these links:

https://www.nltk.org/install.html

https://pypi.org/project/flair/

## Rust AST Parser
To set up the Rust AST parser, follow the instructions at [pyrs_ast/README.md](pyrs_ast/README.md)

## Named-entity Recognition and Semantic Role Labelling
The project is configured to use the NER and SRL tools from the paper `Combining formal and machine learning techniques for the generation of JML specifications`.
[jml_nlp/setup.sh](jml_nlp/setup.sh) will download the tools and set them up using Docker.

### NOTE:
Cite nltk, flair
