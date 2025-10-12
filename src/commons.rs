use anyhow::{Error, Result, anyhow};
use ort::session::Session;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::classification::timm::TimmLike;
use crate::detection::{dfine::DfineLike, yolo::YoloLike};

static DFINE_INPUTS: [&str; 2] = ["images", "orig_target_sizes"];
static DFINE_OUTPUTS: [&str; 3] = ["labels", "boxes", "scores"];
static YOLO_INPUTS: [&str; 1] = ["images"];
static YOLO_OUTPUTS: [&str; 1] = ["output0"];
static TIMM_INPUTS: [&str; 1] = ["input0"];
static TIMM_OUTPUTS: [&str; 1] = ["output0"];

pub enum Provider {
    DfineLike(DfineLike),
    YoloLike(YoloLike),
    TimmLike(TimmLike),
}

pub fn determine_provider(session: Session) -> Result<Provider, Error> {
    let input_names: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
    let output_names: Vec<&str> = session.outputs.iter().map(|o| o.name.as_str()).collect();
    println!("{:?} | {:?}", input_names, output_names);
    if input_names == DFINE_INPUTS && output_names == DFINE_OUTPUTS {
        let (input_width, input_height) = determine_input_shape(&session, "images")?;
        println!(
            "Model is DfineLike with input shape ({},{})",
            input_width, input_height
        );
        Ok(Provider::DfineLike(DfineLike {
            session,
            input_width,
            input_height,
        }))
    } else if input_names == YOLO_INPUTS && output_names == YOLO_OUTPUTS {
        let (input_width, input_height) = determine_input_shape(&session, "images")?;
        println!(
            "Model is YoloLike with input shape ({},{})",
            input_width, input_height
        );
        Ok(Provider::YoloLike(YoloLike {
            session,
            input_width,
            input_height,
        }))
    } else if input_names == TIMM_INPUTS && output_names == TIMM_OUTPUTS {
        let (input_width, input_height) = determine_input_shape(&session, "input0")?;
        println!(
            "Model is TimmLike with input shape ({},{})",
            input_width, input_height
        );
        Ok(Provider::TimmLike(TimmLike {
            session,
            input_width,
            input_height,
        }))
    } else {
        Err(anyhow!("Failed to determine provider!"))
    }
}

pub fn get_onnx_session(path_onnx: &Path) -> Result<Session, Error> {
    let session = Session::builder()?.commit_from_file(path_onnx)?;
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

pub fn determine_input_shape(session: &Session, input_name: &str) -> Result<(u32, u32), Error> {
    println!("{:?}", &session.inputs);
    for input in &session.inputs {
        if input.name == input_name
            && let Some(dims) = input.input_type.tensor_shape() {
                let d = dims.len();
                if d > 1 {
                    let (w, h) = (dims[d - 2], dims[d - 1]);
                    return Ok((w as u32, h as u32));
                }
            }
    }
    Err(anyhow!("Failed to determine input shape!"))
}

pub fn determine_onnx_output(session: &Session) -> Result<(String, u32, u32), Error> {
    println!("{:?}", &session.outputs);
    for output in &session.outputs {
        if let Some(dims) = output.output_type.tensor_shape() {
            let d = dims.len();
            if d > 1 {
                let (w, h) = (dims[d - 2], dims[d - 1]);
                return Ok((String::from(&output.name), w as u32, h as u32));
            }
        }
    }
    Err(anyhow!(
        "Failed to determine ONNX model output - no output tensor found!"
    ))
}
