pub trait SplitFirstChar {
    fn split_first(&self) -> Option<(char, &Self)>;
}

impl SplitFirstChar for str {
    fn split_first(&self) -> Option<(char, &Self)> {
        let mut chars = self.chars();
        let c = chars.next()?;
        Some((c, chars.as_str()))
    }
}
