use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array3, Array4, Axis, s};
use ort::{
    inputs,
    session::{Session, SessionInputValue, SessionOutputs},
};
use std::borrow::Cow;

use crate::detection::{BoundingBox, ObjectDetection, nms};

pub struct YoloLike {
    pub session: Session,
    pub input_width: u32,
    pub input_height: u32,
}

impl ObjectDetection for YoloLike {
    fn make_inputs(
        &self,
        img: &image::DynamicImage,
    ) -> Result<Vec<(Cow<'_, str>, SessionInputValue<'_>)>, Error> {
        let images = img_to_arr(img, self.input_width, self.input_height)?;
        let session_inputs = inputs! {
            "images" => images.view(),
        }?;
        Ok(session_inputs)
    }

    fn make_results(
        &self,
        outputs: SessionOutputs<'_, '_>,
        conf_thres: f32,
        iou_thres: f32,
        max_detect: usize,
    ) -> Result<Vec<BoundingBox>, Error> {
        let output = outputs["output0"].try_extract_tensor::<f32>()?;
        let view_candidates = output.slice(s![0, 4.., ..]);
        let mask_candidates: Vec<bool> = view_candidates
            .axis_iter(Axis(1))
            .map(|col| col.iter().cloned().fold(f32::NEG_INFINITY, f32::max) > conf_thres)
            .collect();
        let idx_candidates: Vec<usize> = mask_candidates
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();
        let candidates_image = output.select(Axis(2), &idx_candidates).squeeze();
        let mut bboxes: Vec<BoundingBox> = Vec::with_capacity(candidates_image.len_of(Axis(1)));
        for candidate in candidates_image.axis_iter(Axis(1)) {
            //println!("\tshape for candidate {:?}: {:?}", idx_candidate, candidate.shape());
            let bbox = BoundingBox::from_array(candidate.to_shape(candidate.len()).unwrap().view());
            bboxes.push(bbox);
        }
        let mut bboxes = nms(&bboxes, iou_thres);
        bboxes.truncate(max_detect); // keep only max detections
        println!("len bboxes nms: {:?}", bboxes.len());
        Ok(bboxes)
    }

    fn run(
        &self,
        img: &DynamicImage,
        conf_thres: f32,
        iou_thres: f32,
        max_detect: usize,
    ) -> Result<Vec<BoundingBox>, Error> {
        let session_inputs = self.make_inputs(img)?;
        let session_outputs = self.session.run(session_inputs)?;
        let mut bboxes = self.make_results(session_outputs, conf_thres, iou_thres, max_detect)?;
        let (base_w, base_h) = (640., 640.);
        let (target_w, target_h) = (img.width() as f32, img.height() as f32);
        let scale_w = target_w / base_w;
        let scale_h = target_h / base_h;
        for bbox in &mut bboxes {
            bbox.scale(scale_w, scale_h);
        }
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
