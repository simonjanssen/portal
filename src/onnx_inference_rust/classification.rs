use anyhow::{Error, Result};
use image::DynamicImage;
use ndarray::{Array1, ArrayView1};
use ort::session::Session;
use ort::session::{SessionInputValue, SessionOutputs};
use std::borrow::Cow;

#[derive(Debug)]
pub struct ClassPrediction {
    pub class_idx: u32,
    pub score: f32,
}

pub trait Classification {
    fn make_inputs(
        &self,
        img: &DynamicImage,
        crop_pct: f32,
    ) -> Result<Vec<(Cow<'_, str>, SessionInputValue<'_>)>, Error>;
    fn make_results(
        &self,
        outputs: SessionOutputs<'_, '_>,
        apply_softmax: bool,
    ) -> Result<Vec<ClassPrediction>, Error>;
    fn run(
        &self,
        session: &Session,
        img: &DynamicImage,
        crop_pct: f32,
        apply_softmax: bool,
    ) -> Result<Vec<ClassPrediction>, Error> {
        let session_inputs = self.make_inputs(img, crop_pct)?;
        let session_outputs = session.run(session_inputs)?;
        self.make_results(session_outputs, apply_softmax)
    }
}

pub fn softmax(input_array: ArrayView1<f32>) -> Result<Array1<f32>, Error> {
    let max_value = input_array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_shifted = input_array.mapv(|x| (x - max_value).exp());
    let sum_exp = exp_shifted.sum();
    Ok(exp_shifted / sum_exp)
}
