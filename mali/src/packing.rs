use anyhow::{ensure, Result};

pub fn transpose_fp16_row_major(source: &[u8], rows: usize, columns: usize) -> Result<Vec<u8>> {
    ensure!(
        source.len() == rows * columns * 2,
        "FP16 matrix byte length mismatch"
    );
    let mut output = vec![0u8; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            let input_offset = (row * columns + column) * 2;
            let output_offset = (column * rows + row) * 2;
            output[output_offset..output_offset + 2]
                .copy_from_slice(&source[input_offset..input_offset + 2]);
        }
    }
    Ok(output)
}
