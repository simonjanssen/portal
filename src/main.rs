use anyhow::{Error, Result, anyhow};
use core::f32;
use image::{DynamicImage, GenericImage};
use image::{GenericImageView, Rgba, imageops::FilterType};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::{Array1, Array3, Array4, ArrayBase, ArrayView1, Axis, s};
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
use ort::inputs;
use ort::session::{Session, SessionOutputs};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use std::env;

const COLORS: [Rgba<u8>; 10] = [
    Rgba([255, 153, 255, 255]), // More intense Magenta
    Rgba([255, 153, 170, 255]), // More intense Pink
    Rgba([255, 204, 153, 255]), // More intense Peach
    Rgba([255, 255, 153, 255]), // More intense Yellow
    Rgba([153, 255, 178, 255]), // More intense Mint Green
    Rgba([153, 204, 255, 255]), // More intense Blue
    Rgba([204, 153, 255, 255]), // More intense Lavender
    Rgba([255, 153, 204, 255]), // More intense Rose
    Rgba([204, 255, 153, 255]), // More intense Lime
    Rgba([153, 255, 255, 255]), // More intense Cyan
];

fn xywh_to_xyxy(x: &f32, y: &f32, w: &f32, h: &f32) -> (f32, f32, f32, f32) {
    let x1 = x - w / 2.0;
    let y1 = y - h / 2.0;
    let x2 = x + w / 2.0;
    let y2 = y + h / 2.0;
    (x1, y1, x2, y2)
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
        Self {
            image,
            target,
            base: (640.0, 640.0),
            classes: None,
            bboxes: None,
        }
    }

    /// loop implementation
    fn into_array_old(&self) -> Result<Array4<f32>, Error> {
        use image::{GenericImageView, Rgba, imageops::FilterType};
        let mut arr4: Array4<f32> = ArrayBase::zeros((1, 3, 640, 640));
        let resized = &self.image.resize_exact(640, 640, FilterType::Triangle);
        for (x, y, Rgba([r, g, b, _])) in resized.pixels() {
            let x = x as usize;
            let y = y as usize;
            arr4[[0, 0, y, x]] = (r as f32) / 255.;
            arr4[[0, 1, y, x]] = (g as f32) / 255.;
            arr4[[0, 2, y, x]] = (b as f32) / 255.;
        }
        println!("image -> array {:?}", arr4.shape());
        Ok(arr4)
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

        //let start = Instant::now();
        //let img_arr_old = self.into_array_old().unwrap();
        //let dt = start.elapsed();
        //println!("[into_array_old] {:?}", dt);

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

    fn annotate(&mut self) -> Result<(), Error> {
        let img_d = self.image.width().min(self.image.height());
        let thickness = 15.0 / 3726. * (img_d as f64); // scale thickness by smaller image edge
        let thickness = (thickness as u32).max(1);
        match &self.bboxes {
            Some(bboxes) => {
                for bbox in bboxes.iter() {
                    let box_color = COLORS[(bbox.class_idx as usize) % COLORS.len()];
                    let (x1, y1, w, h) = bbox.x1y1wh();
                    for t in 0..thickness {
                        let x = x1 - t;
                        let y = y1 - t;
                        let w = w + 2*t;
                        let h = h + 2*t;
                        let rect = Rect::at(x as i32, y as i32).of_size(w as u32, h as u32);
                        draw_hollow_rect_mut(&mut self.image, rect, box_color);
                    }
                }
            }
            _ => {}
        }
        Ok(())
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

fn get_classes(path_json: &Path) -> Result<HashMap<i32, String>, Error> {
    let content = fs::read_to_string(path_json)?;
    let classes: HashMap<String, String> = serde_json::from_str(&content)?;
    let mapping: HashMap<i32, String> = classes
        .into_iter()
        .filter_map(|(k, v)| k.parse::<i32>().ok().map(|ik| (ik, v)))
        .collect();
    Ok(mapping)
}

fn get_onnx_session(path_onnx: &Path) -> Result<Session, Error> {
    let session = Session::builder()?.commit_from_file(&path_onnx)?;
    for input in &session.inputs {
        println!("input name: {:?}", input.name);
    }
    Ok(session)
}

fn run_yolo(model_name: &str, path_image: &str) -> Result<(), Error> {
    ort::init()
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),
        ])
        .commit()?;

    let start = Instant::now();
    let path_onnx = format!("./{}.onnx", model_name);
    let path_json = format!("./{}.json", model_name);
    let session = get_onnx_session(Path::new(&path_onnx))?;
    let classes = get_classes(Path::new(&path_json))?;
    let dt = start.elapsed();
    println!("[session load] {:?}", dt);

    let start = Instant::now();
    let mut object_detection = ObjectDetection::from_path(Path::new(path_image));
    let dt = start.elapsed();
    println!("[model input] {:?}", dt);

    let start = Instant::now();
    object_detection.run(&session, None, None, None)?;
    let dt = start.elapsed();
    println!("[inference] {:?}", dt);

    let start = Instant::now();
    object_detection.save()?;
    let dt = start.elapsed();
    println!("[store results] {:?}", dt);

    let start = Instant::now();
    object_detection.annotate()?;
    object_detection.image.save("./result.png")?;
    let dt = start.elapsed();
    println!("[store results] {:?}", dt);

    Ok(())
}

fn get_timm_image(path_image: &Path) -> Result<Array4<f32>, Error> {
    let start = Instant::now();
    let image_raw = image::ImageReader::open(path_image)?.decode()?;
    let dt = start.elapsed();
    println!("[image load] {:?}", dt);

    let start = Instant::now();
    let resized = image_raw.resize_exact(224, 224, FilterType::Triangle);
    let dt = start.elapsed();
    println!("[resize] {:?}", dt);

    let start = Instant::now();
    let mut tensor: Array4<f32> = ArrayBase::zeros((1, 3, 224, 224));
    for (x, y, Rgba([r, g, b, _])) in resized.pixels() {
        let x = x as usize;
        let y = y as usize;
        tensor[[0, 0, y, x]] = (r as f32) / 255.;
        tensor[[0, 1, y, x]] = (g as f32) / 255.;
        tensor[[0, 2, y, x]] = (b as f32) / 255.;
    }
    let dt = start.elapsed();
    println!("[tensor] {:?}", dt);
    Ok(tensor)
}

fn softmax(input_array: ArrayView1<f32>) -> Result<Array1<f32>, Error> {
    let max_value = input_array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_shifted = input_array.mapv(|x| (x - max_value).exp());
    let sum_exp = exp_shifted.sum();
    Ok(exp_shifted / sum_exp)
}

fn get_timm_predictions(outputs: SessionOutputs) -> Result<(), Error> {
    // extract tensor
    let output = outputs["output0"].try_extract_tensor::<f32>()?;
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
    for (class_idx, prediction) in output.axis_iter(Axis(0)).enumerate() {
        for class_logit in prediction.iter() {
            if *class_logit >= 0.05 {
                println!("class {:?} with prob {:?}", class_idx, class_logit);
            }
        }
    }
    Ok(())
}

fn run_timm() -> Result<(), Error> {
    ort::init()
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),
        ])
        .commit()?;

    let start = Instant::now();
    let path_onnx = Path::new("./mobilenetv3_large_100.onnx");
    let session = get_onnx_session(path_onnx).unwrap();
    let dt = start.elapsed();
    println!("[session load] {:?}", dt);

    let start = Instant::now();
    let path_image = Path::new("./images/bus.jpg");
    let tensor_in = get_timm_image(path_image)?;
    let inputs = inputs!["input0" => tensor_in.view()]?;
    let dt = start.elapsed();
    println!("[model input] {:?}", dt);

    let start = Instant::now();
    let outputs = session.run(inputs)?;
    let dt = start.elapsed();
    println!("[inference] {:?}", dt);

    let start = Instant::now();
    get_timm_predictions(outputs).unwrap();
    let dt = start.elapsed();
    println!("[postprocessing] {:?}", dt);

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("invalid args");
        std::process::exit(1);
    }
    let model_name = &args[1];
    let image_path = &args[2];
    println!("model: {}, image: {}", model_name, image_path);

    run_yolo(model_name, image_path).unwrap();
    //run_timm().unwrap();
    Ok(())
}
