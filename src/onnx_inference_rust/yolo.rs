use anyhow::{Error, Result, anyhow};
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
use std::path::Path;
use std::time::Instant;

use crate::onnx_inference_rust::commons::{get_classes, get_onnx_session};
use crate::onnx_inference_rust::detection::ObjectDetection;

pub fn run_yolo(model_name: &str, path_image: &str) -> Result<(), Error> {
    ort::init()
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),
        ])
        .commit()?;

    let start = Instant::now();
    let path_onnx = format!("./{}.onnx", model_name);
    let path_json = format!("./{}.json", model_name);
    let session = get_onnx_session(Path::new(&path_onnx))?;
    let classes = get_classes(Path::new(&path_json))?;
    let dt = start.elapsed();
    println!("[session load] {:?}", dt);

    let start = Instant::now();
    let mut object_detection = ObjectDetection::from_path(Path::new(path_image));
    let dt = start.elapsed();
    println!("[model input] {:?}", dt);

    let start = Instant::now();
    object_detection.run(&session, None, None, None)?;
    let dt = start.elapsed();
    println!("[inference] {:?}", dt);

    let start = Instant::now();
    object_detection.save()?;
    let dt = start.elapsed();
    println!("[store results] {:?}", dt);

    let start = Instant::now();
    object_detection.annotate()?;
    object_detection.image.save("./result.png")?;
    let dt = start.elapsed();
    println!("[store results] {:?}", dt);

    Ok(())
}
