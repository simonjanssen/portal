use anyhow::{Error, Result, anyhow};
use clap::Parser;
use image::ImageReader;
use onnx_inference_rust::onnx_inference_rust::commons::Provider;
use std::path::Path;

pub use onnx_inference_rust::onnx_inference_rust::classification;
pub use onnx_inference_rust::onnx_inference_rust::commons::{determine_provider, get_onnx_session};
pub use onnx_inference_rust::onnx_inference_rust::detection;
pub use onnx_inference_rust::onnx_inference_rust::dfine::DfineLike;
pub use onnx_inference_rust::onnx_inference_rust::yolo::YoloLike;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    task: String,

    #[arg(long)]
    image: String,

    #[arg(long)]
    model: String,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    let task = args.task;
    let path_onnx = args.model;
    let path_img = args.image;
    println!("model: {}, image: {}", path_onnx, path_img);

    let path_img = Path::new("./images/bus.jpg");
    let img = ImageReader::open(path_img)?.decode()?;

    let session = get_onnx_session(Path::new(&path_onnx))?;
    let provider = determine_provider(&session).ok_or(anyhow!("Unknown Provider!"))?;

    match provider {
        Provider::DfineLike(model) => {}
        Provider::YoloLike(model) => {}
        Provider::TimmLike(model) => {}
    }

    Ok(())
}
