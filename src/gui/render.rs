use crate::simulation::Simulation;
use crate::tilemap::{HEIGHT, WIDTH};
use std::io::{self, Write};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Ppm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldLayerMode {
    Logic,
    Heat,
    Charge,
    HeatAndCharge,
    LogicAndHeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderParams {
    pub fields: FieldLayerMode,
    pub scale: u32, // >= 1
    pub format: ImageFormat,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            fields: FieldLayerMode::Heat,
            scale: 4,
            format: ImageFormat::Ppm,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<RgbPixel>, // row-major, length = width*height
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderError {
    InvalidRegion,
    OutOfBounds,
    InvalidScale,
}

fn clamp_u32_to_u8(x: u32) -> u8 {
    if x > 255 { 255 } else { x as u8 }
}

fn palette_logic(v: u64) -> RgbPixel {
    if v == 0 {
        RgbPixel { r: 0, g: 0, b: 0 }
    } else {
        RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }
    }
}

fn palette_heat(h: u32) -> RgbPixel {
    let r = clamp_u32_to_u8(h);
    RgbPixel { r, g: 0, b: 0 }
}

fn palette_charge(c: u32) -> RgbPixel {
    let b = clamp_u32_to_u8(c);
    RgbPixel { r: 0, g: 0, b }
}

fn blend_add(a: RgbPixel, b: RgbPixel) -> RgbPixel {
    let r = (a.r as u16 + b.r as u16).min(255) as u8;
    let g = (a.g as u16 + b.g as u16).min(255) as u8;
    let b2 = (a.b as u16 + b.b as u16).min(255) as u8;
    RgbPixel { r, g, b: b2 }
}

pub fn render_region(
    sim: &Simulation,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    params: &RenderParams,
) -> Result<ImageBuffer, RenderError> {
    if w == 0 || h == 0 {
        return Err(RenderError::InvalidRegion);
    }
    if params.scale == 0 {
        return Err(RenderError::InvalidScale);
    }
    let x1 = x0.checked_add(w).ok_or(RenderError::OutOfBounds)?;
    let y1 = y0.checked_add(h).ok_or(RenderError::OutOfBounds)?;
    if x1 as usize > WIDTH || y1 as usize > HEIGHT {
        return Err(RenderError::OutOfBounds);
    }

    let out_w = w.saturating_mul(params.scale);
    let out_h = h.saturating_mul(params.scale);
    let mut pixels = vec![RgbPixel { r: 0, g: 0, b: 0 }; (out_w as usize) * (out_h as usize)];

    for (jy, y) in (y0 as usize..y1 as usize).enumerate() {
        for (jx, x) in (x0 as usize..x1 as usize).enumerate() {
            let color = match params.fields {
                FieldLayerMode::Logic => {
                    let v = sim
                        .tilemap
                        .get_tile(x, y)
                        .map(|t| t.logic.load(Ordering::Relaxed))
                        .unwrap_or(0);
                    palette_logic(v)
                }
                FieldLayerMode::Heat => {
                    let hval = sim.get_heat_field_for_test(x, y);
                    palette_heat(hval)
                }
                FieldLayerMode::Charge => {
                    let cval = sim.get_charge_field_for_test(x, y);
                    palette_charge(cval)
                }
                FieldLayerMode::HeatAndCharge => {
                    let hval = sim.get_heat_field_for_test(x, y);
                    let cval = sim.get_charge_field_for_test(x, y);
                    blend_add(palette_heat(hval), palette_charge(cval))
                }
                FieldLayerMode::LogicAndHeat => {
                    let v = sim
                        .tilemap
                        .get_tile(x, y)
                        .map(|t| t.logic.load(Ordering::Relaxed))
                        .unwrap_or(0);
                    let hval = sim.get_heat_field_for_test(x, y);
                    // Heat as base + green logic overlay
                    let base = palette_heat(hval);
                    let overlay = if v == 0 {
                        RgbPixel { r: 0, g: 0, b: 0 }
                    } else {
                        RgbPixel { r: 0, g: 255, b: 0 }
                    };
                    blend_add(base, overlay)
                }
            };

            // Fill scale x scale block
            for sy in 0..params.scale as usize {
                let py = (jy as u32 * params.scale) as usize + sy;
                let row_off = py * out_w as usize;
                let px0 = (jx as u32 * params.scale) as usize;
                for sx in 0..params.scale as usize {
                    pixels[row_off + px0 + sx] = color;
                }
            }
        }
    }

    Ok(ImageBuffer {
        width: out_w,
        height: out_h,
        pixels,
    })
}

pub fn write_ppm(img: &ImageBuffer, mut w: impl Write) -> io::Result<()> {
    // P6 header
    write!(w, "P6\n{} {}\n255\n", img.width, img.height)?;
    // raw data
    for p in &img.pixels {
        w.write_all(&[p.r, p.g, p.b])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;

    #[test]
    fn palette_logic_zero_nonzero() {
        assert_eq!(palette_logic(0), RgbPixel { r: 0, g: 0, b: 0 });
        assert_eq!(
            palette_logic(1),
            RgbPixel {
                r: 255,
                g: 255,
                b: 255
            }
        );
    }

    #[test]
    fn region_validation() {
        let sim = Simulation::new();
        let p = RenderParams::default();
        assert!(matches!(
            render_region(&sim, 0, 0, 0, 1, &p),
            Err(RenderError::InvalidRegion)
        ));
        assert!(matches!(
            render_region(&sim, 0, 0, 1, 0, &p),
            Err(RenderError::InvalidRegion)
        ));
        assert!(matches!(
            render_region(&sim, WIDTH as u32, 0, 1, 1, &p),
            Err(RenderError::OutOfBounds)
        ));
    }

    #[test]
    fn ppm_header_and_size() {
        let sim = Simulation::new();
        let p = RenderParams {
            fields: FieldLayerMode::Heat,
            scale: 2,
            format: ImageFormat::Ppm,
        };
        let img = render_region(&sim, 0, 0, 3, 2, &p).unwrap();
        assert_eq!(img.width, 6);
        assert_eq!(img.height, 4);
        let mut buf = Vec::new();
        write_ppm(&img, &mut buf).unwrap();
        let header = b"P6\n6 4\n255\n";
        assert!(buf.starts_with(header));
        let expected_len = header.len() + (img.width as usize) * (img.height as usize) * 3;
        assert_eq!(buf.len(), expected_len);
    }

    #[test]
    fn deterministic_render() {
        let sim = Simulation::new();
        let p = RenderParams::default();
        let a = render_region(&sim, 1, 1, 2, 2, &p).unwrap();
        let b = render_region(&sim, 1, 1, 2, 2, &p).unwrap();
        assert_eq!(a.pixels, b.pixels);
    }
}
