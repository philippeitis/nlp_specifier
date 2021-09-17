enum Leaf {

}

enum Tree {
    Leaf(String),
    Root(String, Vec<Tree>)
}