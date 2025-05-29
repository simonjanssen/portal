use anyhow::{Error, Result, anyhow};
use image::ImageReader;
use onnx_inference_rust::onnx_inference_rust::commons::ExecutionLogic;
use onnx_inference_rust::onnx_inference_rust::timm::TimmLike;
use ort::session::Session;
use std::env;
use std::path::Path;
use clap::Parser;

pub use onnx_inference_rust::onnx_inference_rust::commons::get_onnx_session;
pub use onnx_inference_rust::onnx_inference_rust::dfine::DfineLike;
pub use onnx_inference_rust::onnx_inference_rust::yolo::YoloLike;
pub use onnx_inference_rust::onnx_inference_rust::classification;
pub use onnx_inference_rust::onnx_inference_rust::detection;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    task: String,

    #[arg(long)]
    image: String,

    #[arg(long)]
    model: String
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

    match task.as_str() {
        "detect" => {
            let provider = detection::determine_provider(&session).unwrap();
            let result = match provider {
                detection::Provider::DfineLike => {
                    let execution = DfineLike {};
                    execution.run(&img, &session)
                },
                detection::Provider::YoloLike => {
                    let execution = YoloLike {};
                    execution.run(&img, &session)
                }
            };
        },
        "classify" => {
            let provider = classification::determine_provider(&session).unwrap();
            let execution = match provider {
                classification::Provider::TimmLike => {
                    let execution = TimmLike {};
                    execution.run(&img, &session)
                },
            };
        },
        _ => {
            return Err(anyhow!("Unknown Task!"))
        }
    }

    Ok(())
}
