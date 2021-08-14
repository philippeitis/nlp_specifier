use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyclass]
struct Token {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    whitespace_after: bool,
    #[pyo3(get)]
    start: usize,
}

#[pyfunction]
#[pyo3(text_signature = "(sentence, /)")]
fn tokens_from_str(s: String) -> Vec<Token> {
    let mut parts = vec![];
    let mut idx = 0;
    let mut start = 0;
    let mut text = String::new();
    let mut char_iter = s.chars().peekable();

    while let Some(c) = char_iter.next() {
        if c.is_whitespace() {
            start += 1;
            continue;
        }

        let mut end_pos = start + 1;
        idx += 1;
        text.push(c);

        match c {
            '\'' | '"' | '`' => while let Some(c2) = char_iter.next() {
                end_pos += 1;
                text.push(c2);

                if c2 == c {
                    parts.push(Token {
                        text: std::mem::take(&mut text),
                        index: idx,
                        whitespace_after: true,
                        start,
                    });
                    // eat quote for next string.
                    start = end_pos;
                    break;
                }
            }
            _ => while let Some(c) = char_iter.next() {
                end_pos += 1;

                if c == ' ' || c == ',' {
                    parts.push(Token {
                        text: std::mem::take(&mut text),
                        index: idx,
                        whitespace_after: true,
                        start,
                    });
                    start = end_pos;
                    break;
                }

                text.push(c);
            }
        }
    }

    if !text.is_empty() {
        parts.push(Token {
            text,
            index: idx,
            whitespace_after: false,
            start,
        });
    }

    parts
}

#[pymodule]
fn doc_parser_utils(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokens_from_str, m)?)?;

    Ok(())
}