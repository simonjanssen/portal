use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array3, Array4, Axis, s};
use ort::inputs;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use std::borrow::Cow;
use std::time::Instant;

use crate::classification::{ClassPrediction, Classification, softmax};

pub struct TimmLike {
    pub session: Session,
    pub input_width: u32,
    pub input_height: u32,
}

impl Classification for TimmLike {
    fn make_inputs(
        &self,
        img: &DynamicImage,
        crop_pct: f32,
    ) -> Result<Vec<(Cow<'_, str>, SessionInputValue<'_>)>, Error> {
        let images = img_to_arr(img, self.input_width, self.input_height, crop_pct)?;
        let session_inputs = inputs! {
            "input0" => images.view(),
        }?;
        Ok(session_inputs)
    }

    fn make_results(
        &self,
        outputs: SessionOutputs<'_, '_>,
        apply_softmax: bool,
    ) -> Result<Vec<ClassPrediction>, Error> {
        let output = outputs["output0"].try_extract_tensor::<f32>()?;
        let output = output.reversed_axes();
        let output = output.slice(s![.., 0]);
        println!("{:?}", output.shape());
        let output = if apply_softmax {
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

    fn run(
        &self,
        img: &DynamicImage,
        crop_pct: f32,
        apply_softmax: bool,
    ) -> Result<Vec<ClassPrediction>, Error> {
        let session_inputs = self.make_inputs(img, crop_pct)?;
        let session_outputs = self.session.run(session_inputs)?;
        self.make_results(session_outputs, apply_softmax)
    }
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
    let arr = {
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
