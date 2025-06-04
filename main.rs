use polars::prelude::*;
use polars::prelude::CsvReader;
use std::fs::File;
use std::error::Error;
use ordered_float::OrderedFloat;
use std::collections::HashSet;
pub fn dist(d:Vec<f64>,q:Vec<f64>) -> Result<f64, String>{
    if d.len() != q.len(){
        return Err("vectors must be same size".to_string());
    }
    let mut dist : f64 = 0.0;
    let length = d.len() - 1;
    for i in 0..length{
    dist += (d[i] - q[i]).powi(2);
    }
    dist = dist.sqrt();
    Ok(dist)
}

pub fn load_data(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let df = CsvReader::new(file)
        .finish()?;

    let height = df.height();
    let width = df.width();

    let mut rows = Vec::with_capacity(height);

    for i in 0..height {
        let row = Vec::with_capacity(width);
        for col in df.get_columns() {
            let val = match col.dtype() {
                DataType::Float64 => col.f64()?.get(i),
                DataType::Int64 => col.i64()?.get(i).map(|v| v as f64),
                DataType::Int32 => col.i32()?.get(i).map(|v| v as f64),
                _ => Some(f64::NAN), // fallback for unsupported types
            };
        }
        rows.push(row);
    }
    Ok(rows)
}
// this will splice the vector from the feature and target value
pub fn partition_nested(nested:Vec<Vec<f64>>) -> (Vec<Vec<f64>>,Vec<f64>){
    let length = nested[0].len() - 1;
    let mut feature_vector : Vec<Vec<f64>> = Vec::new();
    let mut ground_truth = Vec::new();
    // assuming we have preprocessed with target feature at the end of dataframe
    // also assuming categorical targets to use for hashmap
    for arr in nested{
        feature_vector.push(arr[0..length].to_vec());
        let ord = arr[length] as i32;
        ground_truth.push(ord.into());
    }
    (feature_vector,ground_truth)
}


pub fn fit(x: Vec<Vec<f64>>,y:Vec<i32> ,k :usize ,q:Vec<f64>)->Result<i32,String>{
    if k < 3{
        return Err("k must be at least 3".to_string());
    }else if q.len() != x[0].len(){
        return Err("query length must be the same size as feature record".to_string());
    }else if x.is_empty() || y.is_empty(){
        return Err("either feature or target vector is empty".to_string());
    }else{
        let mut counter : usize = 0;
        let mut seen = Vec::new();
        for arr in x{
            let distance = dist(arr,q.clone())?;
            seen.push(vec![OrderedFloat(distance),y[counter].into()]);
            counter += 1; // iterates through all targets
        }
        // sort values to get smallest distances
        seen.sort(); //merge sort O(nlogn)
        let mut set = HashSet::new();
        let mut common = Vec::new();
        for h in 0..k{
            common.push(seen[h][1].into_inner() as i32); // order float must change
            set.insert(seen[h][1].into_inner() as i32);
        }
        let mut res :i32 = 0;
        let mut m :i32 = 0;
        for p in set.iter(){
            let mut count : i32 = 0;
            for a in common.iter(){
                if a == p {
                    count += 1;
                }
            }
            if count > m{
                m = count;
                res = *p;
            }
        }
        Ok(res)
    }
}
fn main() {
    // k-nn usage
    // load csv file , preferably preprocessed for blanks and target features at the end
    // then iterate through result values and fit using new record ,and different k values
}
