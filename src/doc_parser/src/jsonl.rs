use std::io::{Write, BufWriter};
use std::path::Path;
use std::fs::OpenOptions;
use std::collections::HashSet;

use syn::visit::Visit;
use syn::{ItemFn, ImplItemMethod};
use serde::{Serialize, Deserialize};

use crate::docs::Docs;

#[derive(Serialize, Deserialize, Debug, Hash, Eq, PartialEq)]
struct JsonL {
    sentence: String,
    label: Vec<()>,
}

pub struct JsonLValues {
    lines: HashSet<JsonL>,
}

impl<'ast> Visit<'ast> for JsonLValues {
    fn visit_impl_item_method(&mut self, i: &'ast ImplItemMethod) {
        let docs = Docs::from(&i.attrs);
        match docs.sections.first() {
            None => {}
            Some(s) => {
                for sentence in &s.sentences {
                    self.lines.insert(JsonL { sentence: sentence.clone(), label: Vec::new() });
                }
            }
        }
    }

    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        let docs = Docs::from(&i.attrs);
        match docs.sections.first() {
            None => {}
            Some(s) => {
                for sentence in &s.sentences {
                    self.lines.insert(JsonL { sentence: sentence.clone(), label: Vec::new() });
                }
            }
        }
    }
}

impl JsonLValues {
    pub(crate) fn new() -> Self {
        JsonLValues { lines: HashSet::new() }
    }

    pub(crate) fn write_to_path<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut writer = BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(path)?);
        for line in &self.lines {
            writer.write(serde_json::to_string(line).unwrap().as_bytes())?;
            writer.write(b"\n")?;
        }

        Ok(())
    }
}