pub mod timm;

use anyhow::{Error, Result};
use image::DynamicImage;
use ndarray::{Array1, ArrayView1};

#[derive(Debug)]
pub struct ClassPrediction {
    pub class_idx: u32,
    pub score: f32,
}

pub trait Classification {
    fn run(
        &mut self,
        img: &DynamicImage,
        crop_pct: f32,
        apply_softmax: bool,
    ) -> Result<Vec<ClassPrediction>, Error>;
}

pub fn softmax(input_array: ArrayView1<f32>) -> Result<Array1<f32>, Error> {
    let max_value = input_array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_shifted = input_array.mapv(|x| (x - max_value).exp());
    let sum_exp = exp_shifted.sum();
    Ok(exp_shifted / sum_exp)
}
