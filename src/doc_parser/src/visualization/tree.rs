use std::io::Write;
use std::path::Path;
use std::process::Stdio;

use itertools::Itertools;

use crate::parse_tree::{Symbol, SymbolTree};

enum GraphvizSubcommand {
    Dot,
    Circo,
}

impl GraphvizSubcommand {
    fn to_command(&self) -> &'static str {
        match self {
            GraphvizSubcommand::Dot => "dot",
            GraphvizSubcommand::Circo => "circo",
        }
    }
}

struct Counter(usize);

impl Counter {
    fn peek(&self) -> usize {
        self.0
    }

    fn next(&mut self) -> usize {
        self.0 += 1;
        self.0
    }
}

struct Edge {
    color: String,
    arrowhead: String,
    from_id: String,
    to_id: String,
    style: Option<String>,
}

struct Node {
    id: String,
    label: String,
    shape: String,
    color: String,
    fontcolor: String,
}

impl Edge {
    fn prop_str(&self) -> String {
        [
            ("color", Some(&self.color)),
            ("arrowhead", Some(&self.arrowhead)),
            ("style", self.style.as_ref()),
        ]
        .iter()
        .filter_map(|(key, val)| val.map(|val| format!("{}=\"{}\"", key, val)))
        .join(" ")
    }

    fn to_dot(&self) -> String {
        format!("{} -> {} [{}]", self.from_id, self.to_id, self.prop_str())
    }
}

impl Node {
    fn prop_str(&self) -> String {
        [
            ("label", Some(&self.label)),
            ("shape", Some(&self.shape)),
            ("color", Some(&self.color)),
            ("fontcolor", Some(&self.fontcolor)),
        ]
        .iter()
        .filter_map(|(key, val)| val.map(|val| format!("{}=\"{}\"", key, val)))
        .join(" ")
    }

    fn to_dot(&self) -> String {
        format!("{} [{}]", self.id, self.prop_str())
    }
}

enum Item {
    Node(Node),
    Edge(Edge),
}

fn tree_to_graph_helper(
    tree: &SymbolTree,
    counter: &mut Counter,
    leaf_color: Option<&str>,
    items: &mut Vec<Item>,
) {
    let id = counter.peek();
    match tree {
        SymbolTree::Terminal(t) => {
            items.push(Item::Node(Node {
                id: id.to_string(),
                label: t.word.clone(),
                shape: "box".to_string(),
                color: leaf_color.unwrap().to_string(),
                fontcolor: leaf_color.unwrap().to_string(),
            }));
        }
        SymbolTree::Branch(pos, branches) => {
            let color = tag_color(pos);
            items.push(Item::Node(Node {
                id: id.to_string(),
                label: pos.to_string(),
                shape: "none".to_string(),
                color: color.to_string(),
                fontcolor: color.to_string(),
            }));
            for child in branches.iter() {
                let child_id = counter.next();
                let (style, child_color) = match child {
                    SymbolTree::Terminal(_) => (None, color),
                    SymbolTree::Branch(sym, _) => (
                        Some("bold".to_string()),
                        if color == tag_color(sym) {
                            color
                        } else {
                            "#000000"
                        },
                    ),
                };
                items.push(Item::Edge(Edge {
                    color: child_color.to_string(),
                    arrowhead: "none".to_string(),
                    from_id: id.to_string(),
                    to_id: child_id.to_string(),
                    style,
                }));

                tree_to_graph_helper(child, counter, Some(color), items);
            }
        }
    }
}

fn tree_to_graph(tree: &SymbolTree) -> Vec<Item> {
    let mut items = Vec::new();
    tree_to_graph_helper(tree, &mut Counter(0), None, &mut items);
    items
}

pub(crate) fn render_tree<P: AsRef<Path>>(tree: &SymbolTree, path: P) -> std::io::Result<()> {
    let dot_str = tree_to_graph(tree)
        .into_iter()
        .map(|x| match x {
            Item::Node(n) => n.to_dot(),
            Item::Edge(e) => e.to_dot(),
        })
        .join("\n");
    let file_format = match path.as_ref().extension() {
        None => "pdf",
        Some(ext) => ext.to_str().unwrap_or("pdf"),
    };
    let mut cmd = std::process::Command::new(GraphvizSubcommand::Dot.to_command())
        .arg(format!("-T{}", file_format))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    cmd.stdin
        .as_mut()
        .unwrap()
        .write(format!("digraph {{ {}}}\n", dot_str).as_bytes())?;
    println!("digraph {{ {}}}\n", dot_str);
    let output = cmd.wait_with_output().unwrap();
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, output.stdout)
}

pub fn tag_color(sym: &Symbol) -> &'static str {
    use Symbol::*;
    match sym {
        VB | VBP | VBZ | VBN | VBG | VBD | MVB | RET | MRET | RETIF | MD => "#FF8B3D",
        RB => "#ff9a57",
        PRP | NN | NNP | NNS | NNPS | MNN | BOOL_EXPR | CODE | OBJ | LIT => "#00AA00",
        JJ | JJS | JJR | TJJ | MJJ => "#00dd00",
        IN | TO | EQTO | IF | IFF => "#E3242B",
        DT | CC => "#b0b0b0",
        QASSERT | HASSERT | COND => "#E3242B",
        FOR => "#b0b0b0",
        _ => "#000000",
    }
}
// Rendering digraphs for production rules (using circo)

// digraph {
//     rankdir="LR";
//     graph [nodesep="0.1", ranksep="0.5",style="invis"];
//     mindist="0.4";
//     0 [label="OBJ" shape="circle" color="#00AA00" fontcolor="#00AA00"];
//     splines=ortho;
//
//     subgraph cluster1 {
//         rank="same"
//         1 [label="PRP" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         2 [label="(DT)?   MNN" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         3 [label="CODE" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         4 [label="(DT)?   LIT" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         5 [label="OBJ   OP   OBJ" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         6 [label="(DT)?   FNCALL" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         7 [label="DT   VBG   MNN" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//         8 [label="PROP_OF   OBJ" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
//     }
//
//     0 -> {1, 2, 3, 4, 4, 5, 6, 7, 8} [color="#00AA00" arrowhead="normal"];
//
// }
