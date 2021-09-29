#![allow(non_camel_case_types)]
use crate::parse_tree::{Symbol, SymbolTree, Terminal};

#[derive(Clone)]
pub enum S {
    Mret(MRET),
    Retif(RETIF),
    Hassert(HASSERT),
    Qassert(QASSERT),
    Side(SIDE),
    Assign(ASSIGN),
}

impl From<SymbolTree> for S {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for S {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match labels.next() {
            Some((Symbol::MRET, mret_0)) => {
                S::Mret(MRET::from(mret_0))
            },
            Some((Symbol::RETIF, retif_0)) => {
                S::Retif(RETIF::from(retif_0))
            },
            Some((Symbol::HASSERT, hassert_0)) => {
                S::Hassert(HASSERT::from(hassert_0))
            },
            Some((Symbol::QASSERT, qassert_0)) => {
                S::Qassert(QASSERT::from(qassert_0))
            },
            Some((Symbol::SIDE, side_0)) => {
                S::Side(SIDE::from(side_0))
            },
            Some((Symbol::ASSIGN, assign_0)) => {
                S::Assign(ASSIGN::from(assign_0))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum MNN {
    Nn(NN),
    Nns(NNS),
    Nnp(NNP),
    Nnps(NNPS),
    _0(JJ, Box<MNN>),
    _1(VBN, Box<MNN>),
}

impl From<SymbolTree> for MNN {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for MNN {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::NN, nn_0)), None) => {
                MNN::Nn(NN::from(nn_0))
            },
            (Some((Symbol::NNS, nns_0)), None) => {
                MNN::Nns(NNS::from(nns_0))
            },
            (Some((Symbol::NNP, nnp_0)), None) => {
                MNN::Nnp(NNP::from(nnp_0))
            },
            (Some((Symbol::NNPS, nnps_0)), None) => {
                MNN::Nnps(NNPS::from(nnps_0))
            },
            (Some((Symbol::JJ, jj_0)), Some((Symbol::MNN, mnn_1))) => {
                MNN::_0(JJ::from(jj_0), Box::new(MNN::from(mnn_1)))
            },
            (Some((Symbol::VBN, vbn_0)), Some((Symbol::MNN, mnn_1))) => {
                MNN::_1(VBN::from(vbn_0), Box::new(MNN::from(mnn_1)))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum TJJ {
    JJ(Option<DT>, JJ),
    JJR(Option<DT>, JJR),
    JJS(Option<DT>, JJS),
}

impl From<SymbolTree> for TJJ {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for TJJ {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::JJ, jj_1)), None) => {
                TJJ::JJ(None, JJ::from(jj_1))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::JJ, jj_1))) => {
                TJJ::JJ(Some(DT::from(dt_0)), JJ::from(jj_1))
            },
            (Some((Symbol::JJR, jjr_1)), None) => {
                TJJ::JJR(None, JJR::from(jjr_1))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::JJR, jjr_1))) => {
                TJJ::JJR(Some(DT::from(dt_0)), JJR::from(jjr_1))
            },
            (Some((Symbol::JJS, jjs_1)), None) => {
                TJJ::JJS(None, JJS::from(jjs_1))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::JJS, jjs_1))) => {
                TJJ::JJS(Some(DT::from(dt_0)), JJS::from(jjs_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum MJJ {
    JJ(Option<RB>, JJ),
    JJR(Option<RB>, JJR),
    JJS(Option<RB>, JJS),
}

impl From<SymbolTree> for MJJ {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for MJJ {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::JJ, jj_1)), None) => {
                MJJ::JJ(None, JJ::from(jj_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::JJ, jj_1))) => {
                MJJ::JJ(Some(RB::from(rb_0)), JJ::from(jj_1))
            },
            (Some((Symbol::JJR, jjr_1)), None) => {
                MJJ::JJR(None, JJR::from(jjr_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::JJR, jjr_1))) => {
                MJJ::JJR(Some(RB::from(rb_0)), JJR::from(jjr_1))
            },
            (Some((Symbol::JJS, jjs_1)), None) => {
                MJJ::JJS(None, JJS::from(jjs_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::JJS, jjs_1))) => {
                MJJ::JJS(Some(RB::from(rb_0)), JJS::from(jjs_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum MVB {
    VB(Option<RB>, VB),
    VBZ(Option<RB>, VBZ),
    VBP(Option<RB>, VBP),
    VBN(Option<RB>, VBN),
    VBG(Option<RB>, VBG),
    VBD(Option<RB>, VBD),
}

impl From<SymbolTree> for MVB {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for MVB {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::VB, vb_1)), None) => {
                MVB::VB(None, VB::from(vb_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::VB, vb_1))) => {
                MVB::VB(Some(RB::from(rb_0)), VB::from(vb_1))
            },
            (Some((Symbol::VBZ, vbz_1)), None) => {
                MVB::VBZ(None, VBZ::from(vbz_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::VBZ, vbz_1))) => {
                MVB::VBZ(Some(RB::from(rb_0)), VBZ::from(vbz_1))
            },
            (Some((Symbol::VBP, vbp_1)), None) => {
                MVB::VBP(None, VBP::from(vbp_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::VBP, vbp_1))) => {
                MVB::VBP(Some(RB::from(rb_0)), VBP::from(vbp_1))
            },
            (Some((Symbol::VBN, vbn_1)), None) => {
                MVB::VBN(None, VBN::from(vbn_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::VBN, vbn_1))) => {
                MVB::VBN(Some(RB::from(rb_0)), VBN::from(vbn_1))
            },
            (Some((Symbol::VBG, vbg_1)), None) => {
                MVB::VBG(None, VBG::from(vbg_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::VBG, vbg_1))) => {
                MVB::VBG(Some(RB::from(rb_0)), VBG::from(vbg_1))
            },
            (Some((Symbol::VBD, vbd_1)), None) => {
                MVB::VBD(None, VBD::from(vbd_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::VBD, vbd_1))) => {
                MVB::VBD(Some(RB::from(rb_0)), VBD::from(vbd_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum IFF {
    _0(IF, CC, RB, IF),
}

impl From<SymbolTree> for IFF {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for IFF {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::IF, if_0)), Some((Symbol::CC, cc_1)), Some((Symbol::RB, rb_2)), Some((Symbol::IF, if_3))) => {
                IFF::_0(IF::from(if_0), CC::from(cc_1), RB::from(rb_2), IF::from(if_3))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum EQTO {
    _0(IN, CC, JJ, IN),
}

impl From<SymbolTree> for EQTO {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for EQTO {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::IN, in_0)), Some((Symbol::CC, cc_1)), Some((Symbol::JJ, jj_2)), Some((Symbol::IN, in_3))) => {
                EQTO::_0(IN::from(in_0), CC::from(cc_1), JJ::from(jj_2), IN::from(in_3))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum BITOP {
    _0(JJ, CC),
    _1(NN, CC),
}

impl From<SymbolTree> for BITOP {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for BITOP {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::JJ, jj_0)), Some((Symbol::CC, cc_1))) => {
                BITOP::_0(JJ::from(jj_0), CC::from(cc_1))
            },
            (Some((Symbol::NN, nn_0)), Some((Symbol::CC, cc_1))) => {
                BITOP::_1(NN::from(nn_0), CC::from(cc_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum ARITHOP {
    ARITH(ARITH, Option<IN>),
}

impl From<SymbolTree> for ARITHOP {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for ARITHOP {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::ARITH, arith_0)), None) => {
                ARITHOP::ARITH(ARITH::from(arith_0), None)
            },
            (Some((Symbol::ARITH, arith_0)), Some((Symbol::IN, in_1))) => {
                ARITHOP::ARITH(ARITH::from(arith_0), Some(IN::from(in_1)))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum SHIFTOP {
    _0(SHIFT, IN, DT, NN, IN),
    _1(JJ, SHIFT),
}

impl From<SymbolTree> for SHIFTOP {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for SHIFTOP {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::SHIFT, shift_0)), Some((Symbol::IN, in_1)), Some((Symbol::DT, dt_2)), Some((Symbol::NN, nn_3)), Some((Symbol::IN, in_4))) => {
                SHIFTOP::_0(SHIFT::from(shift_0), IN::from(in_1), DT::from(dt_2), NN::from(nn_3), IN::from(in_4))
            },
            (Some((Symbol::JJ, jj_0)), Some((Symbol::SHIFT, shift_1)), None, None, None) => {
                SHIFTOP::_1(JJ::from(jj_0), SHIFT::from(shift_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum OP {
    Bitop(BITOP),
    Arithop(ARITHOP),
    Shiftop(SHIFTOP),
}

impl From<SymbolTree> for OP {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for OP {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match labels.next() {
            Some((Symbol::BITOP, bitop_0)) => {
                OP::Bitop(BITOP::from(bitop_0))
            },
            Some((Symbol::ARITHOP, arithop_0)) => {
                OP::Arithop(ARITHOP::from(arithop_0))
            },
            Some((Symbol::SHIFTOP, shiftop_0)) => {
                OP::Shiftop(SHIFTOP::from(shiftop_0))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum OBJ {
    Prp(PRP),
    MNN(Option<DT>, MNN),
    Code(CODE),
    LIT(Option<DT>, LIT),
    _0(Box<OBJ>, OP, Box<OBJ>),
    Str(STR),
    Char(CHAR),
    _1(DT, VBG, MNN),
    _2(PROP_OF, Box<OBJ>),
}

impl From<SymbolTree> for OBJ {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for OBJ {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::PRP, prp_0)), None, None) => {
                OBJ::Prp(PRP::from(prp_0))
            },
            (Some((Symbol::MNN, mnn_1)), None, None) => {
                OBJ::MNN(None, MNN::from(mnn_1))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::MNN, mnn_1)), None) => {
                OBJ::MNN(Some(DT::from(dt_0)), MNN::from(mnn_1))
            },
            (Some((Symbol::CODE, code_0)), None, None) => {
                OBJ::Code(CODE::from(code_0))
            },
            (Some((Symbol::LIT, lit_1)), None, None) => {
                OBJ::LIT(None, LIT::from(lit_1))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::LIT, lit_1)), None) => {
                OBJ::LIT(Some(DT::from(dt_0)), LIT::from(lit_1))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::OP, op_1)), Some((Symbol::OBJ, obj_2))) => {
                OBJ::_0(Box::new(OBJ::from(obj_0)), OP::from(op_1), Box::new(OBJ::from(obj_2)))
            },
            (Some((Symbol::STR, str_0)), None, None) => {
                OBJ::Str(STR::from(str_0))
            },
            (Some((Symbol::CHAR, char_0)), None, None) => {
                OBJ::Char(CHAR::from(char_0))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::VBG, vbg_1)), Some((Symbol::MNN, mnn_2))) => {
                OBJ::_1(DT::from(dt_0), VBG::from(vbg_1), MNN::from(mnn_2))
            },
            (Some((Symbol::PROP_OF, prop_of_0)), Some((Symbol::OBJ, obj_1)), None) => {
                OBJ::_2(PROP_OF::from(prop_of_0), Box::new(OBJ::from(obj_1)))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum REL {
    _0(TJJ, IN, OBJ),
    _1(TJJ, EQTO, OBJ),
    _2(IN, OBJ),
    _3(Box<REL>, CC, OBJ),
}

impl From<SymbolTree> for REL {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for REL {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::TJJ, tjj_0)), Some((Symbol::IN, in_1)), Some((Symbol::OBJ, obj_2))) => {
                REL::_0(TJJ::from(tjj_0), IN::from(in_1), OBJ::from(obj_2))
            },
            (Some((Symbol::TJJ, tjj_0)), Some((Symbol::EQTO, eqto_1)), Some((Symbol::OBJ, obj_2))) => {
                REL::_1(TJJ::from(tjj_0), EQTO::from(eqto_1), OBJ::from(obj_2))
            },
            (Some((Symbol::IN, in_0)), Some((Symbol::OBJ, obj_1)), None) => {
                REL::_2(IN::from(in_0), OBJ::from(obj_1))
            },
            (Some((Symbol::REL, rel_0)), Some((Symbol::CC, cc_1)), Some((Symbol::OBJ, obj_2))) => {
                REL::_3(Box::new(REL::from(rel_0)), CC::from(cc_1), OBJ::from(obj_2))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum MREL {
    REL(Option<RB>, REL),
}

impl From<SymbolTree> for MREL {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for MREL {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::REL, rel_1)), None) => {
                MREL::REL(None, REL::from(rel_1))
            },
            (Some((Symbol::RB, rb_0)), Some((Symbol::REL, rel_1))) => {
                MREL::REL(Some(RB::from(rb_0)), REL::from(rel_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum PROP {
    _0(MVB, MJJ),
    _1(MVB, MREL),
    _2(MVB, OBJ),
    Mvb(MVB),
    _3(MVB, RANGEMOD),
}

impl From<SymbolTree> for PROP {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for PROP {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::MVB, mvb_0)), Some((Symbol::MJJ, mjj_1))) => {
                PROP::_0(MVB::from(mvb_0), MJJ::from(mjj_1))
            },
            (Some((Symbol::MVB, mvb_0)), Some((Symbol::MREL, mrel_1))) => {
                PROP::_1(MVB::from(mvb_0), MREL::from(mrel_1))
            },
            (Some((Symbol::MVB, mvb_0)), Some((Symbol::OBJ, obj_1))) => {
                PROP::_2(MVB::from(mvb_0), OBJ::from(obj_1))
            },
            (Some((Symbol::MVB, mvb_0)), None) => {
                PROP::Mvb(MVB::from(mvb_0))
            },
            (Some((Symbol::MVB, mvb_0)), Some((Symbol::RANGEMOD, rangemod_1))) => {
                PROP::_3(MVB::from(mvb_0), RANGEMOD::from(rangemod_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum PROP_OF {
    _0(DT, MNN, IN, DT),
    _1(DT, MNN, IN),
    _2(DT, MJJ, IN),
    _3(DT, MJJ, IN, DT),
}

impl From<SymbolTree> for PROP_OF {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for PROP_OF {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::DT, dt_0)), Some((Symbol::MNN, mnn_1)), Some((Symbol::IN, in_2)), Some((Symbol::DT, dt_3))) => {
                PROP_OF::_0(DT::from(dt_0), MNN::from(mnn_1), IN::from(in_2), DT::from(dt_3))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::MNN, mnn_1)), Some((Symbol::IN, in_2)), None) => {
                PROP_OF::_1(DT::from(dt_0), MNN::from(mnn_1), IN::from(in_2))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::MJJ, mjj_1)), Some((Symbol::IN, in_2)), None) => {
                PROP_OF::_2(DT::from(dt_0), MJJ::from(mjj_1), IN::from(in_2))
            },
            (Some((Symbol::DT, dt_0)), Some((Symbol::MJJ, mjj_1)), Some((Symbol::IN, in_2)), Some((Symbol::DT, dt_3))) => {
                PROP_OF::_3(DT::from(dt_0), MJJ::from(mjj_1), IN::from(in_2), DT::from(dt_3))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum RSEP {
    Cc(CC),
    In(IN),
    To(TO),
}

impl From<SymbolTree> for RSEP {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for RSEP {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match labels.next() {
            Some((Symbol::CC, cc_0)) => {
                RSEP::Cc(CC::from(cc_0))
            },
            Some((Symbol::IN, in_0)) => {
                RSEP::In(IN::from(in_0))
            },
            Some((Symbol::TO, to_0)) => {
                RSEP::To(TO::from(to_0))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum RANGE {
    _0(OBJ, IN, OBJ, RSEP, OBJ),
    _1(IN, OBJ, RSEP, OBJ),
    _2(IN, IN, OBJ),
}

impl From<SymbolTree> for RANGE {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for RANGE {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::IN, in_1)), Some((Symbol::OBJ, obj_2)), Some((Symbol::RSEP, rsep_3)), Some((Symbol::OBJ, obj_4))) => {
                RANGE::_0(OBJ::from(obj_0), IN::from(in_1), OBJ::from(obj_2), RSEP::from(rsep_3), OBJ::from(obj_4))
            },
            (Some((Symbol::IN, in_0)), Some((Symbol::OBJ, obj_1)), Some((Symbol::RSEP, rsep_2)), Some((Symbol::OBJ, obj_3)), None) => {
                RANGE::_1(IN::from(in_0), OBJ::from(obj_1), RSEP::from(rsep_2), OBJ::from(obj_3))
            },
            (Some((Symbol::IN, in_0)), Some((Symbol::IN, in_1)), Some((Symbol::OBJ, obj_2)), None, None) => {
                RANGE::_2(IN::from(in_0), IN::from(in_1), OBJ::from(obj_2))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum RANGEMOD {
    Range(RANGE),
    _0(RANGE, Option<COMMA>, JJ),
}

impl From<SymbolTree> for RANGEMOD {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for RANGEMOD {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::RANGE, range_0)), None, None) => {
                RANGEMOD::Range(RANGE::from(range_0))
            },
            (Some((Symbol::RANGE, range_0)), Some((Symbol::JJ, jj_2)), None) => {
                RANGEMOD::_0(RANGE::from(range_0), None, JJ::from(jj_2))
            },
            (Some((Symbol::RANGE, range_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::JJ, jj_2))) => {
                RANGEMOD::_0(RANGE::from(range_0), Some(COMMA::from(comma_1)), JJ::from(jj_2))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum ASSERT {
    _0(OBJ, PROP),
    _1(OBJ, CC, Box<ASSERT>),
}

impl From<SymbolTree> for ASSERT {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for ASSERT {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::PROP, prop_1)), None) => {
                ASSERT::_0(OBJ::from(obj_0), PROP::from(prop_1))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::CC, cc_1)), Some((Symbol::ASSERT, assert_2))) => {
                ASSERT::_1(OBJ::from(obj_0), CC::from(cc_1), Box::new(ASSERT::from(assert_2)))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum HASSERT {
    _0(OBJ, MD, PROP),
    _1(OBJ, CC, Box<HASSERT>),
}

impl From<SymbolTree> for HASSERT {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for HASSERT {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::MD, md_1)), Some((Symbol::PROP, prop_2))) => {
                HASSERT::_0(OBJ::from(obj_0), MD::from(md_1), PROP::from(prop_2))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::CC, cc_1)), Some((Symbol::HASSERT, hassert_2))) => {
                HASSERT::_1(OBJ::from(obj_0), CC::from(cc_1), Box::new(HASSERT::from(hassert_2)))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum QUANT {
    _0(FOR, DT, OBJ),
}

impl From<SymbolTree> for QUANT {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for QUANT {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::FOR, for_0)), Some((Symbol::DT, dt_1)), Some((Symbol::OBJ, obj_2))) => {
                QUANT::_0(FOR::from(for_0), DT::from(dt_1), OBJ::from(obj_2))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum QUANT_EXPR {
    QUANT(QUANT, Option<RANGEMOD>),
    _0(Box<QUANT_EXPR>, Option<COMMA>, CC, MREL),
    _1(Box<QUANT_EXPR>, Option<COMMA>, MREL),
}

impl From<SymbolTree> for QUANT_EXPR {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for QUANT_EXPR {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::QUANT, quant_0)), None, None, None) => {
                QUANT_EXPR::QUANT(QUANT::from(quant_0), None)
            },
            (Some((Symbol::QUANT, quant_0)), Some((Symbol::RANGEMOD, rangemod_1)), None, None) => {
                QUANT_EXPR::QUANT(QUANT::from(quant_0), Some(RANGEMOD::from(rangemod_1)))
            },
            (Some((Symbol::QUANT_EXPR, quant_expr_0)), Some((Symbol::CC, cc_2)), Some((Symbol::MREL, mrel_3)), None) => {
                QUANT_EXPR::_0(Box::new(QUANT_EXPR::from(quant_expr_0)), None, CC::from(cc_2), MREL::from(mrel_3))
            },
            (Some((Symbol::QUANT_EXPR, quant_expr_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::CC, cc_2)), Some((Symbol::MREL, mrel_3))) => {
                QUANT_EXPR::_0(Box::new(QUANT_EXPR::from(quant_expr_0)), Some(COMMA::from(comma_1)), CC::from(cc_2), MREL::from(mrel_3))
            },
            (Some((Symbol::QUANT_EXPR, quant_expr_0)), Some((Symbol::MREL, mrel_2)), None, None) => {
                QUANT_EXPR::_1(Box::new(QUANT_EXPR::from(quant_expr_0)), None, MREL::from(mrel_2))
            },
            (Some((Symbol::QUANT_EXPR, quant_expr_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::MREL, mrel_2)), None) => {
                QUANT_EXPR::_1(Box::new(QUANT_EXPR::from(quant_expr_0)), Some(COMMA::from(comma_1)), MREL::from(mrel_2))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum QASSERT {
    _0(QUANT_EXPR, Option<COMMA>, HASSERT),
    _1(HASSERT, QUANT_EXPR),
    _2(CODE, QUANT_EXPR),
}

impl From<SymbolTree> for QASSERT {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for QASSERT {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::QUANT_EXPR, quant_expr_0)), Some((Symbol::HASSERT, hassert_2)), None) => {
                QASSERT::_0(QUANT_EXPR::from(quant_expr_0), None, HASSERT::from(hassert_2))
            },
            (Some((Symbol::QUANT_EXPR, quant_expr_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::HASSERT, hassert_2))) => {
                QASSERT::_0(QUANT_EXPR::from(quant_expr_0), Some(COMMA::from(comma_1)), HASSERT::from(hassert_2))
            },
            (Some((Symbol::HASSERT, hassert_0)), Some((Symbol::QUANT_EXPR, quant_expr_1)), None) => {
                QASSERT::_1(HASSERT::from(hassert_0), QUANT_EXPR::from(quant_expr_1))
            },
            (Some((Symbol::CODE, code_0)), Some((Symbol::QUANT_EXPR, quant_expr_1)), None) => {
                QASSERT::_2(CODE::from(code_0), QUANT_EXPR::from(quant_expr_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum MRET {
    _0(RET, OBJ),
    _1(OBJ, VBZ, RET),
    _2(OBJ, RET),
}

impl From<SymbolTree> for MRET {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for MRET {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::RET, ret_0)), Some((Symbol::OBJ, obj_1)), None) => {
                MRET::_0(RET::from(ret_0), OBJ::from(obj_1))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::VBZ, vbz_1)), Some((Symbol::RET, ret_2))) => {
                MRET::_1(OBJ::from(obj_0), VBZ::from(vbz_1), RET::from(ret_2))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::RET, ret_1)), None) => {
                MRET::_2(OBJ::from(obj_0), RET::from(ret_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum BOOL_EXPR {
    Assert(ASSERT),
    Qassert(QASSERT),
    Code(CODE),
    Event(EVENT),
}

impl From<SymbolTree> for BOOL_EXPR {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for BOOL_EXPR {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match labels.next() {
            Some((Symbol::ASSERT, assert_0)) => {
                BOOL_EXPR::Assert(ASSERT::from(assert_0))
            },
            Some((Symbol::QASSERT, qassert_0)) => {
                BOOL_EXPR::Qassert(QASSERT::from(qassert_0))
            },
            Some((Symbol::CODE, code_0)) => {
                BOOL_EXPR::Code(CODE::from(code_0))
            },
            Some((Symbol::EVENT, event_0)) => {
                BOOL_EXPR::Event(EVENT::from(event_0))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum COND {
    _0(IF, BOOL_EXPR),
    _1(IFF, BOOL_EXPR),
}

impl From<SymbolTree> for COND {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for COND {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::IF, if_0)), Some((Symbol::BOOL_EXPR, bool_expr_1))) => {
                COND::_0(IF::from(if_0), BOOL_EXPR::from(bool_expr_1))
            },
            (Some((Symbol::IFF, iff_0)), Some((Symbol::BOOL_EXPR, bool_expr_1))) => {
                COND::_1(IFF::from(iff_0), BOOL_EXPR::from(bool_expr_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum RETIF {
    _0(MRET, COND),
    _1(COND, COMMA, MRET),
    _2(Box<RETIF>, COMMA, RB, Box<RETIF>),
    _3(Box<RETIF>, COMMA, RB, OBJ),
    _4(MRET, COMMA, MRET, COND),
}

impl From<SymbolTree> for RETIF {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for RETIF {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::MRET, mret_0)), Some((Symbol::COND, cond_1)), None, None) => {
                RETIF::_0(MRET::from(mret_0), COND::from(cond_1))
            },
            (Some((Symbol::COND, cond_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::MRET, mret_2)), None) => {
                RETIF::_1(COND::from(cond_0), COMMA::from(comma_1), MRET::from(mret_2))
            },
            (Some((Symbol::RETIF, retif_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::RB, rb_2)), Some((Symbol::RETIF, retif_3))) => {
                RETIF::_2(Box::new(RETIF::from(retif_0)), COMMA::from(comma_1), RB::from(rb_2), Box::new(RETIF::from(retif_3)))
            },
            (Some((Symbol::RETIF, retif_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::RB, rb_2)), Some((Symbol::OBJ, obj_3))) => {
                RETIF::_3(Box::new(RETIF::from(retif_0)), COMMA::from(comma_1), RB::from(rb_2), OBJ::from(obj_3))
            },
            (Some((Symbol::MRET, mret_0)), Some((Symbol::COMMA, comma_1)), Some((Symbol::MRET, mret_2)), Some((Symbol::COND, cond_3))) => {
                RETIF::_4(MRET::from(mret_0), COMMA::from(comma_1), MRET::from(mret_2), COND::from(cond_3))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum SIDE {
    _0(OBJ, VBZ, MVB, IN, OBJ),
    _1(OBJ, VBZ, MVB),
    _2(OBJ, VBZ, MJJ, MVB, IN, OBJ),
    _3(VBZ, TJJ, OBJ, IN, OBJ, CC, MRET),
}

impl From<SymbolTree> for SIDE {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for SIDE {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::VBZ, vbz_1)), Some((Symbol::MVB, mvb_2)), Some((Symbol::IN, in_3)), Some((Symbol::OBJ, obj_4)), None, None) => {
                SIDE::_0(OBJ::from(obj_0), VBZ::from(vbz_1), MVB::from(mvb_2), IN::from(in_3), OBJ::from(obj_4))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::VBZ, vbz_1)), Some((Symbol::MVB, mvb_2)), None, None, None, None) => {
                SIDE::_1(OBJ::from(obj_0), VBZ::from(vbz_1), MVB::from(mvb_2))
            },
            (Some((Symbol::OBJ, obj_0)), Some((Symbol::VBZ, vbz_1)), Some((Symbol::MJJ, mjj_2)), Some((Symbol::MVB, mvb_3)), Some((Symbol::IN, in_4)), Some((Symbol::OBJ, obj_5)), None) => {
                SIDE::_2(OBJ::from(obj_0), VBZ::from(vbz_1), MJJ::from(mjj_2), MVB::from(mvb_3), IN::from(in_4), OBJ::from(obj_5))
            },
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::TJJ, tjj_1)), Some((Symbol::OBJ, obj_2)), Some((Symbol::IN, in_3)), Some((Symbol::OBJ, obj_4)), Some((Symbol::CC, cc_5)), Some((Symbol::MRET, mret_6))) => {
                SIDE::_3(VBZ::from(vbz_0), TJJ::from(tjj_1), OBJ::from(obj_2), IN::from(in_3), OBJ::from(obj_4), CC::from(cc_5), MRET::from(mret_6))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum ASSIGN {
    _0(VBZ, OBJ, IN, OBJ),
    _1(VBZ, OBJ, TO, OBJ),
    _2(VBZ, Option<DT>, Option<JJ>, OBJ),
}

impl From<SymbolTree> for ASSIGN {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for ASSIGN {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next(), labels.next(), labels.next()) {
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::OBJ, obj_1)), Some((Symbol::IN, in_2)), Some((Symbol::OBJ, obj_3))) => {
                ASSIGN::_0(VBZ::from(vbz_0), OBJ::from(obj_1), IN::from(in_2), OBJ::from(obj_3))
            },
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::OBJ, obj_1)), Some((Symbol::TO, to_2)), Some((Symbol::OBJ, obj_3))) => {
                ASSIGN::_1(VBZ::from(vbz_0), OBJ::from(obj_1), TO::from(to_2), OBJ::from(obj_3))
            },
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::OBJ, obj_3)), None, None) => {
                ASSIGN::_2(VBZ::from(vbz_0), None, None, OBJ::from(obj_3))
            },
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::DT, dt_1)), Some((Symbol::OBJ, obj_3)), None) => {
                ASSIGN::_2(VBZ::from(vbz_0), Some(DT::from(dt_1)), None, OBJ::from(obj_3))
            },
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::JJ, jj_2)), Some((Symbol::OBJ, obj_3)), None) => {
                ASSIGN::_2(VBZ::from(vbz_0), None, Some(JJ::from(jj_2)), OBJ::from(obj_3))
            },
            (Some((Symbol::VBZ, vbz_0)), Some((Symbol::DT, dt_1)), Some((Symbol::JJ, jj_2)), Some((Symbol::OBJ, obj_3))) => {
                ASSIGN::_2(VBZ::from(vbz_0), Some(DT::from(dt_1)), Some(JJ::from(jj_2)), OBJ::from(obj_3))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub enum EVENT {
    _0(MNN, VBD),
    _1(MNN, VBZ),
}

impl From<SymbolTree> for EVENT {
    fn from(tree: SymbolTree) -> Self {
        let (_symbol, branches) = tree.unwrap_branch();
        Self::from(branches)
    }
}

impl From<Vec<SymbolTree>> for EVENT {
    fn from(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        match (labels.next(), labels.next()) {
            (Some((Symbol::MNN, mnn_0)), Some((Symbol::VBD, vbd_1))) => {
                EVENT::_0(MNN::from(mnn_0), VBD::from(vbd_1))
            },
            (Some((Symbol::MNN, mnn_0)), Some((Symbol::VBZ, vbz_1))) => {
                EVENT::_1(MNN::from(mnn_0), VBZ::from(vbz_1))
            },
            _ => panic!("Unexpected SymbolTree - have you used the code generation with the latest grammar?"),
        }
    }
}

#[derive(Clone)]
pub struct NN {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for NN {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<NN> for Terminal {
    fn from(val: NN) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct NNS {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for NNS {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<NNS> for Terminal {
    fn from(val: NNS) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct NNP {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for NNP {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<NNP> for Terminal {
    fn from(val: NNP) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct NNPS {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for NNPS {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<NNPS> for Terminal {
    fn from(val: NNPS) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct VB {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for VB {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<VB> for Terminal {
    fn from(val: VB) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct VBP {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for VBP {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<VBP> for Terminal {
    fn from(val: VBP) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct VBZ {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for VBZ {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<VBZ> for Terminal {
    fn from(val: VBZ) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct VBN {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for VBN {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<VBN> for Terminal {
    fn from(val: VBN) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct VBG {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for VBG {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<VBG> for Terminal {
    fn from(val: VBG) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct VBD {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for VBD {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<VBD> for Terminal {
    fn from(val: VBD) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct JJ {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for JJ {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<JJ> for Terminal {
    fn from(val: JJ) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct JJR {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for JJR {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<JJR> for Terminal {
    fn from(val: JJR) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct JJS {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for JJS {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<JJS> for Terminal {
    fn from(val: JJS) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct RB {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for RB {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<RB> for Terminal {
    fn from(val: RB) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct PRP {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for PRP {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<PRP> for Terminal {
    fn from(val: PRP) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct DT {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for DT {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<DT> for Terminal {
    fn from(val: DT) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct IN {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for IN {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<IN> for Terminal {
    fn from(val: IN) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct CC {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for CC {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<CC> for Terminal {
    fn from(val: CC) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct MD {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for MD {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<MD> for Terminal {
    fn from(val: MD) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct TO {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for TO {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<TO> for Terminal {
    fn from(val: TO) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct RET {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for RET {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<RET> for Terminal {
    fn from(val: RET) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct CODE {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for CODE {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<CODE> for Terminal {
    fn from(val: CODE) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct LIT {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for LIT {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<LIT> for Terminal {
    fn from(val: LIT) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct IF {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for IF {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<IF> for Terminal {
    fn from(val: IF) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct FOR {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for FOR {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<FOR> for Terminal {
    fn from(val: FOR) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct ARITH {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for ARITH {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<ARITH> for Terminal {
    fn from(val: ARITH) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct SHIFT {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for SHIFT {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<SHIFT> for Terminal {
    fn from(val: SHIFT) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct DOT {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for DOT {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<DOT> for Terminal {
    fn from(val: DOT) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct COMMA {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for COMMA {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<COMMA> for Terminal {
    fn from(val: COMMA) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct EXCL {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for EXCL {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<EXCL> for Terminal {
    fn from(val: EXCL) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct STR {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for STR {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<STR> for Terminal {
    fn from(val: STR) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Clone)]
pub struct CHAR {
    pub word: String,
    pub lemma: String,
}

impl From<Vec<SymbolTree>> for CHAR {
    fn from(mut branches: Vec<SymbolTree>) -> Self {
        let t = branches.remove(0).unwrap_terminal();
        Self {
            word: t.word,
            lemma: t.lemma,
        }
    }
}

impl From<CHAR> for Terminal {
    fn from(val: CHAR) -> Self {
        Self {
            word: val.word,
            lemma: val.lemma,
        }
    }
}

#[derive(Copy, Clone)]
pub enum TerminalSymbol {
    NN,
    NNS,
    NNP,
    NNPS,
    VB,
    VBP,
    VBZ,
    VBN,
    VBG,
    VBD,
    JJ,
    JJR,
    JJS,
    RB,
    PRP,
    DT,
    IN,
    CC,
    MD,
    TO,
    RET,
    CODE,
    LIT,
    IF,
    FOR,
    ARITH,
    SHIFT,
    DOT,
    COMMA,
    EXCL,
    STR,
    CHAR,
}

impl TerminalSymbol {
    pub fn from_terminal<S: AsRef<str>>(s: S) -> Result<Self, String> {
        match s.as_ref() {
            "NN" => Ok(TerminalSymbol::NN),
            "NNS" => Ok(TerminalSymbol::NNS),
            "NNP" => Ok(TerminalSymbol::NNP),
            "NNPS" => Ok(TerminalSymbol::NNPS),
            "VB" => Ok(TerminalSymbol::VB),
            "VBP" => Ok(TerminalSymbol::VBP),
            "VBZ" => Ok(TerminalSymbol::VBZ),
            "VBN" => Ok(TerminalSymbol::VBN),
            "VBG" => Ok(TerminalSymbol::VBG),
            "VBD" => Ok(TerminalSymbol::VBD),
            "JJ" => Ok(TerminalSymbol::JJ),
            "JJR" => Ok(TerminalSymbol::JJR),
            "JJS" => Ok(TerminalSymbol::JJS),
            "RB" => Ok(TerminalSymbol::RB),
            "PRP" => Ok(TerminalSymbol::PRP),
            "DT" => Ok(TerminalSymbol::DT),
            "IN" => Ok(TerminalSymbol::IN),
            "CC" => Ok(TerminalSymbol::CC),
            "MD" => Ok(TerminalSymbol::MD),
            "TO" => Ok(TerminalSymbol::TO),
            "RET" => Ok(TerminalSymbol::RET),
            "CODE" => Ok(TerminalSymbol::CODE),
            "LIT" => Ok(TerminalSymbol::LIT),
            "IF" => Ok(TerminalSymbol::IF),
            "FOR" => Ok(TerminalSymbol::FOR),
            "ARITH" => Ok(TerminalSymbol::ARITH),
            "SHIFT" => Ok(TerminalSymbol::SHIFT),
            "." => Ok(TerminalSymbol::DOT),
            "DOT" => Ok(TerminalSymbol::DOT),
            "," => Ok(TerminalSymbol::COMMA),
            "COMMA" => Ok(TerminalSymbol::COMMA),
            "!" => Ok(TerminalSymbol::EXCL),
            "EXCL" => Ok(TerminalSymbol::EXCL),
            "STR" => Ok(TerminalSymbol::STR),
            "CHAR" => Ok(TerminalSymbol::CHAR),
            x => Err(format!("Terminal {} is not supported.", x)),
        }
    }
}
