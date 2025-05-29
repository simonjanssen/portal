use anyhow::{Error, Result};
use clap::Parser;
use image::ImageReader;
use onnx_inference_rust::onnx_inference_rust::commons::Provider;
use onnx_inference_rust::onnx_inference_rust::detection::ObjectDetection;
use std::path::Path;

pub use onnx_inference_rust::onnx_inference_rust::classification::Classification;
pub use onnx_inference_rust::onnx_inference_rust::commons::{determine_provider, get_onnx_session};
pub use onnx_inference_rust::onnx_inference_rust::detection;
pub use onnx_inference_rust::onnx_inference_rust::dfine::DfineLike;
pub use onnx_inference_rust::onnx_inference_rust::yolo::YoloLike;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    image: String,

    #[arg(long)]
    model: String,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    let path_onnx = args.model;
    let path_img = args.image;
    println!("model: {}, image: {}", path_onnx, path_img);

    let path_img = Path::new("./images/bus.jpg");
    let img = ImageReader::open(path_img)?.decode()?;

    let session = get_onnx_session(Path::new(&path_onnx))?;
    let provider = determine_provider(&session)?;

    match provider {
        Provider::DfineLike(model) => {
            let prediction = model.run(&session, &img, 0.25, 0.7, 300)?;
            println!("{:?}", prediction.len())
        }
        Provider::YoloLike(model) => {
            let prediction = model.run(&session, &img, 0.25, 0.7, 300)?;
            println!("{:?}", prediction.len())
        }
        Provider::TimmLike(model) => {
            let prediction = model.run(&session, &img, 0.875, true)?;
            println!("{:?}", prediction.len())
        }
    }

    Ok(())
}
