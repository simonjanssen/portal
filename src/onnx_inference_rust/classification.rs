use anyhow::{Error, Result, anyhow};
use image::ImageReader;
use image::{DynamicImage, GenericImageView, Rgba, imageops::FilterType};
use ndarray::{Array1, Array3, Array4, ArrayBase, ArrayView1, Axis, s};
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
use ort::inputs;
use ort::session::{Session, SessionOutputs, input};
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::onnx_inference_rust::commons::{
    determine_onnx_input, determine_onnx_output, get_onnx_session,
};

#[derive(Debug)]
pub struct ClassProb {
    class_idx: u32,
    class_prob: f32,
}

/// # DynamicImage to ONNX Input Tensor
/// 1. Center-crop for ONNX model input size
/// 2. Scale between 0..1
/// 3. Normalize every channel by mean and std
///
/// https://github.com/huggingface/pytorch-image-models/blob/main/onnx_validate.py
fn img_to_arr(
    img: &DynamicImage,
    arr_width: u32,
    arr_height: u32,
    crop_pct: f32,
) -> Result<Array4<f32>, Error> {
    let arr_width_f = arr_width as f32;
    let arr_height_f = arr_height as f32;

    let resize_width = arr_width_f / crop_pct;
    let resize_height = arr_height_f / crop_pct;
    println!("{:?}, {:?}", resize_width, resize_height);

    let (img_width, img_height) = img.dimensions(); 

    let resize_width = if img_width > img_height {
        resize_width * (img_width as f32 / img_height as f32)
    } else {
        resize_width
    };

    let resize_height = if img_height > img_width {
        resize_height * ( img_height as f32 / img_width as f32)
    } else {
        resize_height
    };

    //let resize_width = 256.;
    //let resize_height = 256.;

    let x = (resize_width - arr_width_f) / 2.0;
    let y = (resize_height - arr_height_f) / 2.0;

    // first resize, then crop a centered square from resized such that cropped/resized = crop_pct and cropped = ONNX input shape
    println!("{:?}", img.dimensions());
    let img_resized = img
        .resize(
            resize_width as u32,
            resize_height as u32,
            FilterType::Triangle,
        );
    println!("{:?}", img_resized.dimensions());
    img_resized.save("./resized.jpg")?;

    let img_cropped = img_resized
        .crop_imm(x as u32, y as u32, arr_width, arr_height);
    println!("{:?}", img_cropped.dimensions());
    img_cropped.save("./cropped.jpg")?;
    
    let buf_u8 = img_cropped
        .to_rgb8()
        .into_raw();

    // normalize image
    let mut buf_f32 = Vec::with_capacity(buf_u8.len());
    let mean_rgb = [0.4850, 0.4560, 0.4060];
    let std_rgb = [0.2290, 0.2240, 0.2250];
    for (i, &v) in buf_u8.iter().enumerate() {
        let channel = i % 3;
        let norm_val = (((v as f32) / 255.0) - mean_rgb[channel]) / std_rgb[channel];
        buf_f32.push(norm_val);
    }

    // reshape to 3d-array
    let arr4 = Array3::from_shape_vec((arr_height as usize, arr_width as usize, 3), buf_f32)?
        .permuted_axes([2, 0, 1])
        .insert_axis(Axis(0));
    Ok(arr4)
}

pub fn softmax(input_array: ArrayView1<f32>) -> Result<Array1<f32>, Error> {
    let max_value = input_array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_shifted = input_array.mapv(|x| (x - max_value).exp());
    let sum_exp = exp_shifted.sum();
    Ok(exp_shifted / sum_exp)
}

pub fn get_timm_predictions(outputs: SessionOutputs, output_name: &str) -> Result<(), Error> {
    // extract tensor
    let output = outputs[output_name].try_extract_tensor::<f32>()?;
    println!("{:?}", output.shape());

    // flip axes
    let output = output.reversed_axes();
    println!("{:?}", output.shape());

    // squeeze
    let output = output.slice(s![.., 0]);
    println!("{:?}", output.shape());

    // apply softmax
    let output = softmax(output)?;
    println!("softmax verify: {:?}", output.sum());

    // iterate over all classes

    let mut predictions = output
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(class_idx,class_prob)| ClassProb { class_idx: class_idx as u32, class_prob: class_prob.first().unwrap_or(&0.).clone()})
        .collect::<Vec<ClassProb>>();

    predictions.sort_by(| a, b| b.class_prob.partial_cmp(&a.class_prob).unwrap());

    for prediction in predictions.iter().take(5) {
        println!("{:?}", &prediction);
    }

    // tensor([874, 654, 705, 779, 920]) tensor([0.5545, 0.3265, 0.0868, 0.0023, 0.0015], grad_fn=<TopkBackward0>)

    Ok(())
}

pub fn run_timm(model_name: &str, path_image: &str) -> Result<(), Error> {
    ort::init()
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),
        ])
        .commit()?;

    let start = Instant::now();
    let path_onnx = format!("./{}", model_name);
    let session = get_onnx_session(Path::new(&path_onnx))?;
    let (input_name, input_width, input_height) = determine_onnx_input(&session)?;
    let output_name = determine_onnx_output(&session)?;
    let dt = start.elapsed();
    println!("[session load] {:?}", dt);

    let start = Instant::now();
    let path_image = Path::new(path_image);
    let img = ImageReader::open(path_image)?.decode()?;
    let tensor_in = img_to_arr(&img, input_width, input_height, 0.875)?;
    let inputs = inputs![&input_name => tensor_in.view()]?;
    let dt = start.elapsed();
    println!("[model input] {:?}", dt);

    let start = Instant::now();
    let outputs = session.run(inputs)?;
    let dt = start.elapsed();
    println!("[inference] {:?}", dt);

    let start = Instant::now();
    get_timm_predictions(outputs, &output_name).unwrap();
    let dt = start.elapsed();
    println!("[postprocessing] {:?}", dt);

    Ok(())
}
