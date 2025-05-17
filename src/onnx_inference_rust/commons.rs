use anyhow::{Error, Result, anyhow};
use image::Rgba;
use ort::session::Session;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const COLORS: [Rgba<u8>; 10] = [
    Rgba([255, 153, 255, 255]), // More intense Magenta
    Rgba([255, 153, 170, 255]), // More intense Pink
    Rgba([255, 204, 153, 255]), // More intense Peach
    Rgba([255, 255, 153, 255]), // More intense Yellow
    Rgba([153, 255, 178, 255]), // More intense Mint Green
    Rgba([153, 204, 255, 255]), // More intense Blue
    Rgba([204, 153, 255, 255]), // More intense Lavender
    Rgba([255, 153, 204, 255]), // More intense Rose
    Rgba([204, 255, 153, 255]), // More intense Lime
    Rgba([153, 255, 255, 255]), // More intense Cyan
];

pub fn get_onnx_session(path_onnx: &Path) -> Result<Session, Error> {
    let session = Session::builder()?.commit_from_file(&path_onnx)?;
    Ok(session)
}

pub fn get_classes(path_json: &Path) -> Result<HashMap<i32, String>, Error> {
    let content = fs::read_to_string(path_json)?;
    let classes: HashMap<String, String> = serde_json::from_str(&content)?;
    let mapping: HashMap<i32, String> = classes
        .into_iter()
        .filter_map(|(k, v)| k.parse::<i32>().ok().map(|ik| (ik, v)))
        .collect();
    Ok(mapping)
}

/// Determine input-tensor name and shape to resize our images accordingly
/// For simplicity, we are assuming that the first tensor is the image-related one.
pub fn determine_onnx_input(session: &Session) -> Result<(String, u32, u32), Error> {
    for input in &session.inputs {
        if let Some(dims) = input.input_type.tensor_dimensions() {
            let d = dims.len();
            if d > 1 {
                let (w, h) = (dims[d - 2], dims[d - 1]);
                return Ok((String::from(&input.name), w as u32, h as u32));
            }
        }
    }
    Err(anyhow!(
        "Failed to determine ONNX model input - no input tensor found!"
    ))
}

/// Determine output-tensor name to extract as array
/// For simplicity, we are assuming that the first tensor is the prediction-related one.
pub fn determine_onnx_output(session: &Session) -> Result<String, Error> {
    for output in &session.outputs {
        if let Some(dims) = output.output_type.tensor_dimensions() {
            let d = dims.len();
            if d > 1 {
                let (w, h) = (dims[d - 2], dims[d - 1]);
                return Ok(String::from(&output.name));
            }
        }
    }
    Err(anyhow!(
        "Failed to determine ONNX model output - no output tensor found!"
    ))
}
