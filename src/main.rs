use anyhow::{Error, Result};
use clap::Parser;
use image::ImageReader;
use portal::classification::Classification;
use portal::commons::Provider;
use std::path::Path;

use portal::commons::{determine_provider, get_onnx_session};
use portal::detection::ObjectDetection;
use portal::visualize::draw_bboxes;

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
    let provider = determine_provider(session)?;

    match provider {
        Provider::DfineLike(model) => {
            let prediction = model.run(&img, 0.40, 0.7, 300)?;
            println!("{:?}", prediction.len());
            let annotated = draw_bboxes(img, &prediction)?;
            annotated.save("./result.jpg")?;
        }
        Provider::YoloLike(model) => {
            let prediction = model.run(&img, 0.25, 0.7, 300)?;
            println!("{:?}", prediction.len());
            let annotated = draw_bboxes(img, &prediction)?;
            annotated.save("./result.jpg")?;
        }
        Provider::TimmLike(model) => {
            let prediction = model.run(&img, 0.875, true)?;
            println!("{:?}", prediction.len())
        }
    }

    Ok(())
}
