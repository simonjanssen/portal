use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView, Rgba, imageops::FilterType};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ndarray::{Array1, Array3, Array4, ArrayBase, ArrayView1, Axis, s};
use ort::inputs;
use ort::session::Session;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use ab_glyph::{Font, FontArc, ScaleFont};

use crate::onnx_inference_rust::commons::COLORS;
use crate::onnx_inference_rust::commons::get_classes;

pub struct ObjectDetection {
    pub image: DynamicImage,
    pub classes: Option<HashMap<i32, String>>,
    pub bboxes: Option<Vec<BoundingBox>>,
    pub base: (f32, f32),
    pub target: (f32, f32),
}

impl ObjectDetection {
    pub fn from_path(path_image: &Path) -> Self {
        let image = image::ImageReader::open(path_image)
            .unwrap()
            .decode()
            .unwrap();
        let target = (image.width() as f32, image.height() as f32);
        //let path_classes = Path::new("./checkpoints/yolo_v11/yolo11n.json");
        //let classes = get_classes(path_classes).unwrap();
        Self {
            image,
            target,
            base: (640.0, 640.0),
            classes: None,
            bboxes: None,
        }
    }

    /// zero-copy implementation
    fn into_array(&self) -> Result<Array4<f32>, Error> {
        let (width, height) = (640, 640);
        let buf_u8 = self
            .image
            .resize_exact(width, height, FilterType::Triangle)
            .to_rgb8()
            .into_raw();

        let buf_f32: Vec<f32> = buf_u8.into_iter().map(|v| (v as f32) / 255.0).collect();

        let arr4 = Array3::from_shape_vec((height as usize, width as usize, 3), buf_f32)
            .expect("buffer length mismatch") // todo: don't panic
            .permuted_axes([2, 0, 1])
            .insert_axis(Axis(0));

        println!("image -> array {:?}", arr4.shape());
        Ok(arr4)
    }

    pub fn run(
        &mut self,
        session: &Session,
        iou_thres: Option<f32>,
        conf_thres: Option<f32>,
        max_detect: Option<u32>,
    ) -> Result<(), Error> {
        let iou_thres = iou_thres.unwrap_or(0.7);
        let conf_thres = conf_thres.unwrap_or(0.25);
        let max_detect = max_detect.unwrap_or(300);

        let start = Instant::now();
        let img_arr = self.into_array().unwrap();
        let dt = start.elapsed();
        println!("[into_array] {:?}", dt);

        let inputs = {
            match inputs!["images" => img_arr.view()] {
                Ok(mapping) => mapping,
                Err(e) => panic!("todo"),
            }
        };
        let outputs = session.run(inputs)?;

        // extract tensor
        let output = outputs["output0"].try_extract_tensor::<f32>()?; // assumes ONNX-model output is [BATCH_DIM, ]
        println!("{:?}", output.shape());

        let view_candidates = output.slice(s![0, 4.., ..]);
        println!("view candidates: {:?}", output.shape());

        // determine candidates for which the max over all class conf is > conf_thres
        let mask_candidates: Vec<bool> = view_candidates
            .axis_iter(Axis(1))
            .map(|col| col.iter().cloned().fold(f32::NEG_INFINITY, f32::max) > conf_thres)
            .collect();
        println!("mask_candidates: {:?}", mask_candidates.len());

        // get candidate rows
        let idx_candidates: Vec<usize> = mask_candidates
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();
        println!("idx_candidates: {:?}", idx_candidates.len());

        // select candidates = all detections with at least one class conf > conf_thres
        let candidates_image = output.select(Axis(2), &idx_candidates).squeeze(); // todo: handle batch processing
        println!("candidates_image: {:?}", candidates_image.shape());

        // extract bboxes from output vectors
        let mut bboxes: Vec<BoundingBox> = Vec::with_capacity(candidates_image.len_of(Axis(1)));
        for (idx_candidate, candidate) in candidates_image.axis_iter(Axis(1)).enumerate() {
            //println!("\tshape for candidate {:?}: {:?}", idx_candidate, candidate.shape());
            let bbox = BoundingBox::from_array(candidate.to_shape(candidate.len()).unwrap().view());
            bboxes.push(bbox);
        }
        println!("len bboxes: {:?}", bboxes.len());

        // apply nms
        let mut bboxes = nms(&bboxes, iou_thres);
        bboxes.truncate(max_detect as usize); // keep only max detections
        println!("len bboxes nms: {:?}", bboxes.len());

        // scale boxes to original image dims
        let (base_w, base_h) = self.base;
        let (target_w, target_h) = self.target;
        let scale_w = target_w / base_w;
        let scale_h = target_h / base_h;
        for bbox in &mut bboxes {
            bbox.scale(scale_w, scale_h);
        }
        self.bboxes = Some(bboxes);

        Ok(())
    }

    pub fn save(&self) -> Result<(), Error> {
        match &self.bboxes {
            Some(bboxes) => {
                for (b, bbox) in bboxes.iter().enumerate() {
                    let (x, y, w, h) = bbox.x1y1wh();
                    let cropped = &self.image.crop_imm(x, y, w, h);
                    let filename = format!("{:?}.png", b);
                    let path = Path::new(&filename);
                    cropped.save(path)?;
                }
            }
            None => {}
        }
        Ok(())
    }

    pub fn annotate(&mut self) -> Result<(), Error> {
        let img_d = self.image.width().min(self.image.height());
        let thickness = 15.0 / 3726. * (img_d as f32); // scale thickness by smaller image edge
        let thickness = (thickness as u32).max(1);
        let font_data = include_bytes!("../../assets/DejaVuSans.ttf");
        let font = FontArc::try_from_slice(font_data as &[u8]).unwrap();
        let scale = 100.0 / 3726. * (img_d as f32);
        let offset = (scale * 1.1) as u32;
        match &self.bboxes {
            Some(bboxes) => {
                for bbox in bboxes.iter() {
                    let box_color = COLORS[(bbox.class_idx as usize) % COLORS.len()];
                    let (x1, y1, w, h) = bbox.x1y1wh();
                    for t in 0..thickness {
                        let x = x1 - t;
                        let y = y1 - t;
                        let w = w + 2 * t;
                        let h = h + 2 * t;
                        let rect = Rect::at(x as i32, y as i32).of_size(w as u32, h as u32);
                        draw_hollow_rect_mut(&mut self.image, rect, box_color);
                    }

                    let label = match &self.classes {
                        Some(classes) => {
                            format!("{} ({:.2})", classes[&bbox.class_idx], bbox.score)
                        },
                        _ => {
                            format!("class {} ({:.2})", bbox.class_idx, bbox.score)
                        }
                    };
                    println!("{}", label);
                    draw_text_mut(
                        &mut self.image, 
                        box_color, 
                        x1 as i32, 
                        (y1 - offset) as i32, 
                        scale, 
                        &font, 
                        &label
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32, // left
    pub y1: f32, // top
    pub x2: f32, // right
    pub y2: f32, // bottom
    pub score: f32,
    pub class_idx: i32,
}

impl BoundingBox {
    /// center-x, center-y, width, height
    pub fn xywh(&self) -> (u32, u32, u32, u32) {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        let x = (self.x2 + self.x1) / 2.0;
        let y = (self.y2 + self.y1) / 2.0;
        (x as u32, y as u32, w as u32, h as u32)
    }

    /// left, top, width, height
    pub fn x1y1wh(&self) -> (u32, u32, u32, u32) {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        (self.x1 as u32, self.y1 as u32, w as u32, h as u32)
    }

    pub fn area(&self) -> f32 {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        if w > 0.0 && h > 0.0 { w * h } else { 0.0 }
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1_inter = self.x1.max(other.x1);
        let y1_inter = self.y1.max(other.y1);
        let x2_inter = self.x2.min(other.x2);
        let y2_inter = self.y2.min(other.y2);

        let w_inter = x2_inter - x1_inter;
        let h_inter = y2_inter - y1_inter;

        let intersection = if w_inter > 0.0 && h_inter > 0.0 {
            w_inter * h_inter
        } else {
            0.0
        };

        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    pub fn from_array(array: ArrayView1<f32>) -> Self {
        let bbox_xywh = array.slice(s![..4]).to_vec();
        let confs = array.slice(s![4..]).to_vec();
        let (class_idx, conf) = confs
            .iter()
            .enumerate()
            .filter_map(
                |(idx, &num)| {
                    if num.is_nan() { None } else { Some((idx, num)) }
                },
            )
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .unwrap();
        let (x1, y1, x2, y2) =
            xywh_to_xyxy(&bbox_xywh[0], &bbox_xywh[1], &bbox_xywh[2], &bbox_xywh[3]);
        Self {
            x1,
            y1,
            x2,
            y2,
            score: conf,
            class_idx: class_idx as i32,
        }
    }

    pub fn scale(&mut self, scale_w: f32, scale_h: f32) {
        self.x1 *= scale_w;
        self.y1 *= scale_h;
        self.x2 *= scale_w;
        self.y2 *= scale_h;
    }
}

/// Class-Sensitive Non Maxima Suppression for Overlapping Bounding Boxes
/// Iteratively removes lower scoring bboxes which have an IoU above iou_thresold.
/// Inspired by: https://pytorch.org/vision/master/_modules/torchvision/ops/boxes.html#nms
pub fn nms(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Compute the maximum coordinate value among all boxes
    let max_coordinate = boxes.iter().fold(0.0_f32, |max_coord, bbox| {
        max_coord.max(bbox.x2).max(bbox.y2)
    });
    let offset = max_coordinate + 1.0;

    // Create a vector of shifted boxes with their original indices
    let mut boxes_shifted: Vec<(BoundingBox, usize)> = boxes
        .iter()
        .enumerate()
        .map(|(i, bbox)| {
            let class_offset = offset * bbox.class_idx as f32;
            let shifted_bbox = BoundingBox {
                x1: bbox.x1 + class_offset,
                y1: bbox.y1 + class_offset,
                x2: bbox.x2 + class_offset,
                y2: bbox.y2 + class_offset,
                score: bbox.score,
                class_idx: bbox.class_idx, // Keep class_idx the same
            };
            (shifted_bbox, i) // Keep track of the original index
        })
        .collect();

    // Sort boxes in decreasing order based on scores
    boxes_shifted
        .sort_unstable_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap_or(Ordering::Equal));

    let mut keep_indices = Vec::new();

    while let Some((current_box, original_index)) = boxes_shifted.first().cloned() {
        keep_indices.push(original_index);
        boxes_shifted.remove(0);

        // Retain boxes that have an IoU less than or equal to the threshold with the current box
        boxes_shifted.retain(|(bbox, _)| current_box.iou(bbox) <= iou_threshold);
    }

    // Collect the kept boxes from the original input
    let mut kept_boxes: Vec<BoundingBox> = keep_indices
        .into_iter()
        .map(|idx| boxes[idx].clone())
        .collect();

    // Sort the kept boxes in decreasing order of their scores
    kept_boxes.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    kept_boxes
}

fn xywh_to_xyxy(x: &f32, y: &f32, w: &f32, h: &f32) -> (f32, f32, f32, f32) {
    let x1 = x - w / 2.0;
    let y1 = y - h / 2.0;
    let x2 = x + w / 2.0;
    let y2 = y + h / 2.0;
    (x1, y1, x2, y2)
}
