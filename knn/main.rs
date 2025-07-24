use std::collections::HashSet;
use ordered_float::NotNan;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::error::Error;

struct KNNRegression;

impl KNNRegression{
    pub fn euc_distance(&self,u:Vec<f64>,v:Vec<f64>) -> Result<f64,String>{
        let mut dist : f64 = 0.0;
        let length : usize = u.len();
        if length != v.len(){
            return Err("Length is not equal".to_string());
        }else{
            for i in 0..length{
                dist += (u[i] - v[i]).powi(2);
            }
        }
        let res : f64 = dist.sqrt();
        Ok(res)
    }
    pub fn fit(&self,X:&Vec<Vec<f64>>,Y:&Vec<f64>,k:usize,dist_weighted:bool) -> f64{
        let data_length : usize = X.len();
        let vector_length : usize = X[0].len();
        let mut k_closest : Vec<(f64,usize)> = Vec::new();
        for i in 0..data_length{
            let attr : Vec<f64> = X[i][0..vector_length - 2].to_vec();
            let vector : Vec<f64> = Y[0..vector_length - 2].to_vec();
            let r_dist = self.euc_distance(attr,vector).unwrap();
            k_closest.push((r_dist,i));
        }
        k_closest.sort_by(|a,b| a.partial_cmp(b).unwrap()); // sort to find smallest distance
        let mut final_value : f64 = 0.0;
        for j in 0..k{
            let last_value : f64 = X[k_closest[j].1][vector_length - 1];
            if dist_weighted == true{
                if k_closest[j].0 <= 0.0{ // avoid zero weights
                    continue
                }
                final_value += (1.0 / k_closest[j].0) * last_value;
            }else{
                final_value += (1.0 / k as f64) * last_value;
            }
        }
        final_value
    }
    pub fn sse(&self,x:f64,y:&f64) -> f64{
        let res : f64 = (x - y).powi(2);
        res
    }
    pub fn predict(&self,X:Vec<Vec<f64>>,Y:Vec<Vec<f64>>,k:usize,dist_weight:bool) -> f64{
        let test_length : usize = Y.len();
        let vector_length : usize = Y[0].len();
        let mut total_error : f64 = 0.0;
        for i in 0..test_length{
            let prediction = self.fit(&X,&Y[i],k,dist_weight);
            let error : f64 = self.sse(prediction,&Y[i][vector_length - 2]);
            println!("Vector Error ,{:?}",error);
            total_error += error;
        }
        println!("Total Model Error {:?}",total_error);
        total_error
    }
}
//define a struct for our csv file
// using Deserialize in serde
#[derive(Debug, Deserialize)]
struct CancerCsv{
    SMOKING:f64,
    YELLOW_FINGERS:f64,
    ANXIETY:f64,
    PEER_PRESSURE:f64,
    CHRONICDISEASE:f64,
    FATIGUE:f64 ,
    ALLERGY:f64 ,
    WHEEZING:f64,
    ALCOHOLCONSUMING:f64,
    COUGHING:f64,
    SHORTNESSOFBREATH:f64,
    SWALLOWINGDIFFICULTY:f64,
    CHESTPAIN:f64,
    LUNG_CANCER:f64,
}
impl CancerCsv {
    fn make_vec(&self) -> Vec<f64> {
        vec![self.SMOKING, self.YELLOW_FINGERS, self.ANXIETY, self.PEER_PRESSURE, self.CHRONICDISEASE, self.FATIGUE, self.ALLERGY, self.WHEEZING,
             self.ALCOHOLCONSUMING, self.COUGHING, self.SHORTNESSOFBREATH, self.SWALLOWINGDIFFICULTY, self.CHESTPAIN, self.LUNG_CANCER]
    }
    // we have to manually crate a vec for the struct
}
    fn load_csv(file:&str) -> Result<Vec<Vec<f64>>,Box<dyn Error>>{
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file)?;
        let mut rows : Vec<Vec<f64>> = Vec::new();

        for result in rdr.deserialize(){
            let row : CancerCsv = result?;
            let act_row : Vec<f64> = row.make_vec();
            rows.push(act_row);
        }
        Ok(rows)
    }



    
fn main(){
    let model = KNNRegression;
    let train = load_csv("src/train.csv").expect("File Parsing Failed");
    let test = load_csv("src/test.csv").expect("File Parsing Failed");
    let err = model.predict(train,test,5,false);
}
 

