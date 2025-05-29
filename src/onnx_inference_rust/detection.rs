use ab_glyph::{FontArc, ScaleFont};
use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ndarray::{ArrayView1, s};
use ort::session::Session;
use std::cmp::Ordering;

use crate::onnx_inference_rust::commons::COLORS;

use super::commons::ExecutionLogic;
use super::dfine::DfineLike;
use super::yolo::YoloLike;

// manually determined scale factors to print annotations / draw boxes
const SCALE_THICKNESS: f32 = 15. / 3726.;
const SCALE_FONT: f32 = 100. / 3726.;

pub type Detector = dyn ExecutionLogic<Prediction = Vec<BoundingBox>>;

static DFINE_INPUTS: [&str; 2] = ["images", "orig_target_sizes"];
static DFINE_OUTPUTS: [&str; 3] = ["labels", "boxes", "scores"];

static YOLO_INPUTS: [&str; 1] = ["images"];
static YOLO_OUTPUTS: [&str; 1] = ["output0"];

pub enum Provider {
    DfineLike(DfineLike),
    YoloLike(YoloLike),
}

impl Provider {
    fn run(img: ) {
        match self {
            Provider::DfineLike(dfine) => {
                dfine.run(img, session)
            }
        }
    }
}

pub fn determine_provider(session: &Session) -> Option<Provider> {
    let input_names: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
    let output_names: Vec<&str> = session.outputs.iter().map(|o| o.name.as_str()).collect();
    println!("{:?} | {:?}", input_names, output_names);
    if input_names == DFINE_INPUTS && output_names == DFINE_OUTPUTS {
        Some(Provider::DfineLike)
    } else if input_names == YOLO_INPUTS && output_names == YOLO_OUTPUTS {
        Some(Provider::YoloLike)
    } else {
        None
    }
}

fn xywh_to_xyxy(x: &f32, y: &f32, w: &f32, h: &f32) -> (f32, f32, f32, f32) {
    let x1 = x - w / 2.0;
    let y1 = y - h / 2.0;
    let x2 = x + w / 2.0;
    let y2 = y + h / 2.0;
    (x1, y1, x2, y2)
}

#[derive(Default, Clone, Debug, Copy)]
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

/// # Draw Rectangles
/// Draws hollow rectangles onto input image using BoundingBox coordinates
/// Applies box thickness that is dynamically scaled by input image resolution
fn draw_bboxes(mut img: DynamicImage, bboxes: &Vec<BoundingBox>) -> Result<DynamicImage, Error> {
    let img_d = img.width().min(img.height()) as f32;
    let thickness = SCALE_THICKNESS * img_d; // scale thickness by smaller image edge
    let thickness = (thickness as u32).max(1);

    let font_data = include_bytes!("../../assets/DejaVuSans.ttf");
    let font = FontArc::try_from_slice(font_data as &[u8]).unwrap();
    let font_scale = SCALE_FONT * img_d;
    let font_offset = (font_scale * 1.1) as u32;

    for bbox in bboxes.iter() {
        let box_color = COLORS[(bbox.class_idx as usize) % COLORS.len()];
        let (x1, y1, w, h) = bbox.x1y1wh();
        for t in 0..thickness {
            let x = x1 - t;
            let y = y1 - t;
            let w = w + 2 * t;
            let h = h + 2 * t;
            let rect = Rect::at(x as i32, y as i32).of_size(w, h);
            draw_hollow_rect_mut(&mut img, rect, box_color);

            let label = format!("class {} ({:.2})", bbox.class_idx, bbox.score);
            draw_text_mut(
                &mut img,
                box_color,
                x1 as i32,
                (y1 - font_offset) as i32,
                font_scale,
                &font,
                &label,
            );
        }
    }
    Ok(img)
}
