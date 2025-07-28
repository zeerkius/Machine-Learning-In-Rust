use rand::Rng;
use std::collections::HashMap;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::error::Error;

struct KMeans;
impl KMeans{
    fn euc_distance(&self,u:&Vec<f64>,v:&Vec<f64>) -> Result<f64,String> {
        let mut distance: f64 = 0.0;
        let k: usize = u.len();
        if k != v.len() {
            return Err("Vectors must be the same size".to_string());
        } else {
            for i in 0..k {
                distance += (u[i] - v[i]).powi(2);
            }
            let dist: f64 = distance.sqrt();
            Ok(dist)
        }
    }    
    fn sse(&self,x:f64,y:f64) -> f64{
        let err : f64 = (x - y).powi(2);
        err
    }
    fn gen_centroid(&self,l:usize,max:f64,min:f64) -> Vec<f64>{
        let mut centroid : Vec<f64> = Vec::new();
        let mut rnd = rand::thread_rng(); // inst of rand
        for i in 0..l{
            let random : f64 = rnd.gen_range(min..max); // we use the min and max of the dataset to param
            centroid.push(random);
        }
        centroid
    }
    fn get_max(&self,X:&Vec<f64>) -> f64{
        X.iter().cloned().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap()
    }
    fn get_min(&self,X:&Vec<f64>) -> f64{
        X.iter().cloned().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap()
    }
    fn check_nan(&self,X:&Vec<f64>) -> bool{
        let size : usize = X.len();
        for i in 0..size{
            if X[i].is_nan() == true{
                return true
            }
        }
        false
    }
    fn get_mean(&self,X:&Vec<f64>) -> f64{
        let size : usize = X.len();
        let sum : f64 = X.iter().sum();
        let mean : f64 = (sum / size as f64);
        mean
    }
    fn transpose_avg(&self,X:&Vec<Vec<f64>>) -> Vec<f64>{
        let mut T : Vec<Vec<f64>> = Vec::new();
        let mut res: Vec<f64> = Vec::new();
        if let Some(row_len) = X.first().map(|row| row.len()){
            T = (0..row_len).map(|i| X.iter().map(|row| row[i]).collect()).collect();
        }
        for column in T{
            res.push(self.get_mean(&column));
        }
        res
    }
    // we use max-by min-by to operate and unwrap since we know data shouldn't be Nan
    fn fit(&self,X:Vec<Vec<f64>>,k:usize,epochs:usize) -> Result<HashMap<usize,Vec<Vec<f64>>>,String>{
        let adj_map : HashMap<usize,Vec<Vec<f64>>> =  HashMap::new();
        if X.is_empty(){
            return Err("Data is empty".to_string());
        }
        let mut centroids : Vec<Vec<f64>> = Vec::new();
        let flatten_data : Vec<f64> = X.clone().into_iter().flatten().collect(); // we will use X later
        if self.check_nan(&flatten_data) == true{
            return Err("Data contains NaN value".to_string());
        }
        // we need to get around NaN
        let max = self.get_max(&flatten_data);
        let min = self.get_min(&flatten_data);
        let vec_size : usize = X[0].len();
        let data_size : usize = X.len();
        for i in 0..k{
            let cent = self.gen_centroid(vec_size,max,min);
            centroids.push(cent);
        }
        for i in 0..epochs{
            let mut adj_map : HashMap<usize,Vec<Vec<f64>>> =  HashMap::new(); // for every epoch
            // insertion based on centroid index 
            for j in 0..data_size{
                let mut closest : Vec<f64> = Vec::new();
                for h in 0..centroids.len(){
                    let distance_to_centroid : f64 = self.euc_distance(&centroids[h],&X[j]).unwrap();
                    closest.push(distance_to_centroid);
                }
                // (key -> index ,  value -> vec_closest to that centroid's index)
                let smallest_dist : f64 = self.get_min(&closest);
                let smallest_index : usize = closest.iter().position(|x| *x == smallest_dist).unwrap(); // first occurence
                adj_map.entry(smallest_index).or_default().push(X[j].to_vec());
            }
            //  change the centroids
            for (key, value) in adj_map.iter(){
                centroids[*key] = self.transpose_avg(&value);// taking the mean of all the values
            }
        }
        // after clustering we give the data the label based on the index
        // this is intended to be purley a place-holder and changed if needed by the user
        Ok(adj_map)
    }
}


fn main() {
    
}
