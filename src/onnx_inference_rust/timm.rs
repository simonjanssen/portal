use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array1, Array3, Array4, ArrayView1, Axis, s};
use ort::inputs;
use ort::session::SessionOutputs;
use std::time::Instant;

use super::classification::ClassPrediction;
use super::commons::ExecutionLogic;

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
    let (img_width, img_height) = img.dimensions();
    let buf_u8 = if (img_width == arr_width) && (img_height == arr_height) {
        img.to_rgb8().into_raw()
    } else {
        let arr_width_f = arr_width as f32;
        let arr_height_f = arr_height as f32;

        let resize_width = arr_width_f / crop_pct;
        let resize_height = arr_height_f / crop_pct;

        let resize_width = if img_width > img_height {
            resize_width * (img_width as f32 / img_height as f32)
        } else {
            resize_width
        };

        let resize_height = if img_height > img_width {
            resize_height * (img_height as f32 / img_width as f32)
        } else {
            resize_height
        };
        println!("{:?}, {:?}", resize_width, resize_height);

        //let resize_width = 256.;
        //let resize_height = 256.;

        let x = (resize_width - arr_width_f) / 2.0;
        let y = (resize_height - arr_height_f) / 2.0;

        // first resize, then crop a centered square from resized such that cropped/resized = crop_pct and cropped = ONNX input shape
        println!("{:?}", img.dimensions());
        let img_resized = img.resize(
            resize_width as u32,
            resize_height as u32,
            FilterType::CatmullRom,
        );
        println!("{:?}", img_resized.dimensions());
        img_resized.save("./resized.jpg")?;

        let img_cropped = img_resized.crop_imm(x as u32, y as u32, arr_width, arr_height);
        println!("{:?}", img_cropped.dimensions());
        img_cropped.save("./cropped.jpg")?;

        img_cropped.into_rgb8().into_raw()
    };

    let mean_rgb = [0.4850, 0.4560, 0.4060];
    let std_rgb = [0.2290, 0.2240, 0.2250];

    // normalize image
    let start = Instant::now();
    let arr = if false {
        println!("--- for ---");
        let mut buf_f32 = Vec::with_capacity(buf_u8.len());

        for (i, &v) in buf_u8.iter().enumerate() {
            let channel = i % 3;
            let norm_val = (((v as f32) / 255.0) - mean_rgb[channel]) / std_rgb[channel];
            buf_f32.push(norm_val);
        }

        // reshape to 3d-array
        Array3::from_shape_vec((arr_height as usize, arr_width as usize, 3), buf_f32)?
            .permuted_axes([2, 0, 1])
            .insert_axis(Axis(0))
    } else {
        let buf_f32: Vec<f32> = buf_u8.iter().map(|&v| (v as f32) / 255.0).collect();
        let arr3 = Array3::from_shape_vec((arr_height as usize, arr_width as usize, 3), buf_f32)?;

        // Normalize per channel
        let mut arr3 = arr3; // make mutable
        for c in 0..3 {
            arr3.slice_mut(s![.., .., c]).map_inplace(|x| {
                *x = (*x - mean_rgb[c]) / std_rgb[c];
            });
        }

        arr3.permuted_axes([2, 0, 1]).insert_axis(Axis(0))
    };
    let dt = start.elapsed();
    println!("[normalize] {:?}", dt);
    Ok(arr)
}

pub fn softmax(input_array: ArrayView1<f32>) -> Result<Array1<f32>, Error> {
    let max_value = input_array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_shifted = input_array.mapv(|x| (x - max_value).exp());
    let sum_exp = exp_shifted.sum();
    Ok(exp_shifted / sum_exp)
}

pub struct TimmLike {
    crop_pct: f32,
    apply_softmax: bool,
}

impl ExecutionLogic for TimmLike {
    type Prediction = Vec<ClassPrediction>;
    fn make_inputs(
        &self,
        img: &DynamicImage,
    ) -> Result<
        Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )>,
        Error,
    > {
        let images = img_to_arr(img, 224, 224, self.crop_pct)?;
        let session_inputs = inputs! {
            "images" => images.view(),
        }?;
        Ok(session_inputs)
    }

    fn make_results(&self, outputs: SessionOutputs<'_, '_>) -> Result<Self::Prediction, Error> {
        let output = outputs["output"].try_extract_tensor::<f32>()?;
        println!("{:?}", output.shape());
        let output = output.reversed_axes();
        println!("{:?}", output.shape());
        let output = output.slice(s![.., 0]);
        println!("{:?}", output.shape());
        let output = if self.apply_softmax {
            softmax(output)?
        } else {
            output.to_owned()
        };
        let mut predictions = Vec::with_capacity(output.len_of(Axis(0)));
        for (class_idx, score) in output.axis_iter(Axis(0)).enumerate() {
            let score = score.first().copied().unwrap_or(0.);
            predictions.push(ClassPrediction {
                class_idx: class_idx as u32,
                score,
            });
        }
        predictions.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(predictions)
    }
}
