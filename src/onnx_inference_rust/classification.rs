use super::commons::ExecutionLogic;
use crate::onnx_inference_rust::timm::TimmLike;
use ort::session::Session;

pub type Classifier = dyn ExecutionLogic<Prediction = Vec<ClassPrediction>>;

#[derive(Debug)]
pub struct ClassPrediction {
    pub class_idx: u32,
    pub score: f32,
}

static TIMM_INPUTS: [&str; 2] = ["images", "orig_target_sizes"];
static TIMM_OUTPUTS: [&str; 3] = ["labels", "boxes", "scores"];

pub enum Provider {
    TimmLike,
}

pub fn determine_provider(session: &Session) -> Option<Provider> {
    let input_names: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
    let output_names: Vec<&str> = session.outputs.iter().map(|o| o.name.as_str()).collect();
    println!("{:?} | {:?}", input_names, output_names);
    if input_names == TIMM_INPUTS && output_names == TIMM_OUTPUTS {
        Some(Provider::TimmLike)
    } else {
        None
    }
}
