use std::fmt::{Display, Formatter};
use syn::Attribute;

#[derive(Default)]
pub(crate) struct RawSection {
    pub header: Option<String>,
    pub lines: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Section {
    pub header: Option<String>,
    lines: Vec<String>,
    pub body: String,
    pub sentences: Vec<String>,
}

pub fn split_string_on(s: &str, c: char) -> Vec<&str> {
    // TODO: Fix abbreviation error cases (or use spacy E2E?)
    match c {
        '"' | '`' | '\'' => panic!("Bad char"),
        _ => {}
    }

    let mut looking_for = None;
    let mut splits = vec![];
    let mut prev_ind = 0;

    let mut char_inds = s.char_indices().peekable();
    while let Some((ind, cx)) = char_inds.next() {
        match looking_for {
            Some(')') => match cx {
                '"' | '`' | '\'' => while let Some((_, cx_)) = char_inds.next() {
                    if cx_ == cx {
                        break;
                    }
                },
                ')' => looking_for = None,
                _ => continue,
            },
            Some(x) if x == cx => {
                looking_for = None
            }
            Some(_) => continue,
            None => match cx {
                '"' | '`' | '\'' => looking_for = Some(cx),
                '(' => looking_for = Some(')'),
                _ => {
                    if cx == c {
                        splits.push(s[prev_ind..ind].trim());
                        prev_ind = ind + 1;
                    }
                }
            },
        }
    }

    splits.push(s[prev_ind..s.len()].trim());
    splits.into_iter().filter(|s| s.len() != 0).collect()
}

impl RawSection {
    pub fn new(header: Option<String>) -> Self {
        RawSection {
            header,
            lines: Vec::new(),
        }
    }

    pub fn push_line(&mut self, line: String) {
        self.lines.push(line);
    }
}

impl From<RawSection> for Section {
    fn from(raw_sec: RawSection) -> Self {
        let body = raw_sec.lines.join(" ");
        let sentences = split_string_on(&body, '.')
            .into_iter()
            .map(String::from)
            .collect();
        Section {
            header: raw_sec.header,
            lines: raw_sec.lines,
            body,
            sentences,
        }
    }
}

#[derive(Default)]
pub struct RawDocs {
    sections: Vec<RawSection>,
}

impl RawDocs {
    pub fn push_line<S: AsRef<str>>(&mut self, line: S) {
        let line = line.as_ref();
        let line_strip = line.trim_matches(|c| c == ' ' || c == '\"');
        if line_strip.starts_with("#") {
            self.sections.push(RawSection::new(Some(line.to_string())));
        } else {
            if self.sections.is_empty() {
                self.sections.push(RawSection::new(None));
            }
            self.sections
                .last_mut()
                .unwrap()
                .push_line(line_strip.to_string())
        }
    }

    pub fn push_lines(&mut self, lines: String) {
        lines.lines().for_each(|line| self.push_line(line));
    }

    pub fn consolidate(self) -> Docs {
        Docs::from(self)
    }
}

#[derive(Clone, Debug)]
pub struct Docs {
    pub sections: Vec<Section>,
}

impl From<RawDocs> for Docs {
    fn from(raw_docs: RawDocs) -> Self {
        Docs {
            sections: raw_docs.sections.into_iter().map(Section::from).collect(),
        }
    }
}

impl<A: AsRef<[Attribute]>> From<A> for Docs {
    fn from(attrs: A) -> Self {
        if attrs.as_ref().is_empty() {
            return RawDocs::default().consolidate();
        }

        let mut raw_docs = RawDocs::default();
        for attr in attrs.as_ref() {
            let path = &attr.path;
            if quote::quote!(#path).to_string() == "doc" {
                let tokens = &attr.tokens;
                let s = quote::quote!(#tokens).to_string();
                let k = s.strip_prefix("= \"").unwrap().strip_suffix("\"").unwrap();
                raw_docs.push_line(k);
            }
        }
        raw_docs.consolidate()
    }
}

impl Display for Docs {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut is_first = true;
        for section in &self.sections {
            if let Some(header) = &section.header {
                if is_first {
                    write!(f, "/// {}", header)?;
                    is_first = false;
                } else {
                    write!(f, "\n/// {}", header)?;
                }
            }
            for line in &section.lines {
                if is_first {
                    write!(f, "/// {}", line)?;
                    is_first = false;
                } else {
                    write!(f, "\n/// {}", line)?;
                }
            }
        }
        Ok(())
    }
}
