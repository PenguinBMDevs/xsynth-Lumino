use super::{Sf2ParseError, Sf2SampleLinkType};
use crate::resample::resample_vec;
use rayon::prelude::*;
use soundfont::raw::{SampleChunk, SampleData, SampleHeader};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct Sf2Sample {
    pub data: Arc<[f32]>,
    pub link_type: Sf2SampleLinkType,
    pub linked_sample: Option<u16>,
    pub original_length: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub sample_rate: u32,
    pub origpitch: u8,
    pub pitchadj: i8,
}

impl Sf2Sample {
    fn read_chunk(file: &mut File, chunk: SampleChunk) -> io::Result<Vec<u8>> {
        let mut buff = vec![0; chunk.len as usize];

        file.seek(SeekFrom::Start(chunk.offset))?;
        file.read_exact(&mut buff)?;

        Ok(buff)
    }

    pub fn parse_sf2_samples(
        file: &mut File,
        headers: Vec<SampleHeader>,
        data: SampleData,
        sample_rate: u32,
    ) -> Result<Vec<Self>, Sf2ParseError> {
        let smpl = if let Some(chunk) = data.smpl {
            Self::read_chunk(file, chunk).map_err(|_| {
                Sf2ParseError::FailedToParseFile("Error reading sample contents".to_string())
            })?
        } else {
            return Err(Sf2ParseError::FailedToParseFile(
                "Soundfont does not contain samples".to_string(),
            ));
        };

        let mut samples = Vec::new();

        if let Some(sm24) = data.sm24 {
            // SF2 is 24-bit
            let extra = Self::read_chunk(file, sm24).map_err(|_| {
                Sf2ParseError::FailedToParseFile("Error reading extra sample contents".to_string())
            })?;

            let smpllen = smpl.len() / 2;
            let extralen = extra.len() - (smpllen % 2);
            if smpllen != extralen {
                return Err(Sf2ParseError::FailedToParseFile(
                    "Invalid sample length".to_string(),
                ));
            }

            for i in 0..extralen {
                let n0 = extra[i];
                let n1 = smpl[i * 2];
                let n2 = smpl[i * 2 + 1];
                let sign = if (n2 & 0x80) != 0 { 0xFF } else { 0x00 };
                let sample = i32::from_le_bytes([n0, n1, n2, sign]);
                let conv = sample as f32 / 8_388_607.0;
                samples.push(conv);
            }
        } else {
            // SF2 is 16-bit
            for i in smpl.chunks(2) {
                let n0 = i[0];
                let n1 = i[1];
                let sample = i16::from_le_bytes([n0, n1]);
                let conv = sample as f32 / i16::MAX as f32;
                samples.push(conv);
            }
        }

        let out: Vec<Sf2Sample> = headers
            .into_par_iter()
            .map(|h| {
                let start = h.start;
                let end = h.end;
                let sample: Vec<f32> = samples[start as usize..end as usize].into();

                Sf2Sample {
                    data: if h.sample_rate != sample_rate && !sample.is_empty() {
                        resample_vec(sample, h.sample_rate as f32, sample_rate as f32)
                    } else {
                        sample.into()
                    },
                    link_type: h.sample_type.into(),
                    linked_sample: match h.sample_type.into() {
                        Sf2SampleLinkType::Mono => None,
                        _ => Some(h.sample_link),
                    },
                    original_length: end - start,
                    loop_start: h.loop_start - start,
                    loop_end: h.loop_end - start,
                    sample_rate: h.sample_rate,
                    origpitch: h.origpitch,
                    pitchadj: h.pitchadj,
                }
            })
            .collect();

        Ok(out)
    }
}
