use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array2, Array3, Array4, Axis, s};
use ort::inputs;
use ort::session::{SessionInputValue, SessionOutputs};
use std::borrow::Cow;

use super::detection::{BoundingBox, ObjectDetection};

pub struct DfineLike {
    pub input_width: u32,
    pub input_height: u32,
}

impl ObjectDetection for DfineLike {
    fn make_inputs(
        &self,
        img: &DynamicImage,
    ) -> Result<Vec<(Cow<'_, str>, SessionInputValue<'_>)>, Error> {
        let (img_width, img_height) = (img.width() as i64, img.height() as i64);
        let images = img_to_arr(img, self.input_width, self.input_height)?;
        let orig_target_size = Array2::from_shape_vec((1, 2), vec![img_width, img_height])?;
        let session_inputs = inputs! {
            "images" => images.view(),
            "orig_target_sizes" => orig_target_size.view()
        }?;
        Ok(session_inputs)
    }

    fn make_results(
        &self,
        outputs: SessionOutputs<'_, '_>,
        conf_thres: f32,
        _iou_thres: f32,
        max_detect: usize,
    ) -> Result<Vec<BoundingBox>, Error> {
        let labels = outputs["labels"].try_extract_tensor::<i64>()?;
        let boxes = outputs["boxes"].try_extract_tensor::<f32>()?;
        let scores = outputs["scores"].try_extract_tensor::<f32>()?;
        let mut bboxes: Vec<BoundingBox> = boxes
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(i, bbox)| {
                let bbox_xyxy = bbox.slice(s![0, ..]).to_vec();
                let (x1, y1, x2, y2) = (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]);
                let class_idx = labels.slice(s![.., i]).to_vec()[0];
                let score = scores.slice(s![.., i]).to_vec()[0];
                BoundingBox {
                    class_idx: class_idx as i32,
                    score,
                    x1,
                    y1,
                    x2,
                    y2,
                }
            })
            .filter(|b| b.score > conf_thres)
            .collect();
        bboxes.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bboxes.truncate(max_detect);
        Ok(bboxes)
    }
}

fn img_to_arr(img: &DynamicImage, width: u32, height: u32) -> Result<Array4<f32>, Error> {
    let (img_width, img_height) = img.dimensions();

    let buf_u8 = if (img_width == width) && (img_height == height) {
        img.to_rgb8().into_raw()
    } else {
        img.resize_exact(width, height, FilterType::Triangle)
            .into_rgb8()
            .into_raw()
    };

    // to float tensor
    let buf_f32: Vec<f32> = buf_u8.into_iter().map(|v| (v as f32) / 255.0).collect();

    // expand into 4dim array
    let arr4 = Array3::from_shape_vec((height as usize, width as usize, 3), buf_f32)?
        .permuted_axes([2, 0, 1])
        .insert_axis(Axis(0));
    Ok(arr4)
}
