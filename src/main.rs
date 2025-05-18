use anyhow::Result;
use std::env;

pub use onnx_inference_rust::onnx_inference_rust::classification::run_timm;
pub use onnx_inference_rust::onnx_inference_rust::yolo::run_yolo;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("invalid args");
        std::process::exit(1);
    }
    let model_name = &args[1];
    let path_image = &args[2];
    println!("model: {}, image: {}", model_name, path_image);

    //run_yolo(model_name, path_image).unwrap();
    run_timm(model_name, path_image)?;
    Ok(())
}
