use std::collections::HashSet;
use ordered_float::NotNan;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::error::Error;
struct Data{
    XTrain : Vec<Vec<f64>>,
    XTest : Vec<Vec<f64>>,
}


struct NaiveBayes;
impl NaiveBayes {
    // this implementation assumes features are either continuous or categorical , and if categorical in ordinal form all fo f64 type
    // it returns outputs or True Class and  False Class where we let true == 1, false == 0;
    fn mean(&self, X: &Vec<f64>) -> f64 {
        let mut res: f64 = 0.0;
        let length: usize = X.len();
        for i in 0..length {
            res += X[i];
        }
        res = res / length as f64;
        res
    }
    fn st_deviation(&self, X: &Vec<f64>, mu: f64) -> f64{
        let length: usize = X.len();
        let mut std: f64 = 0.0;
        for j in 0..length {
            let diff: f64 = (X[j] - mu).powi(2);
            std += diff;
        }
        std
    }
    fn norm_pdf(&self, X: &Vec<f64>, x: f64) -> f64 {
        let mut res: f64 = 0.0;
        let mu: f64 = self.mean(&X);
        let std: f64 = self.st_deviation(&X, mu);
        let constant: f64 = (1.0 / 2.50662827463);
        let e: f64 = 2.71828182846;
        let exp: f64 = (-(mu - x).powi(2)) / (2.0 * std.powi(2));
        res = constant * e.powf(exp);
        res
    }
    fn split_class(&self, X: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // splitting into either class
        X.into_iter().partition(|row| row.last() == Some(&0.0))
    }
    fn prod_vec(&self,X:Vec<f64>) -> f64{
        let mut res : f64 = 1.0;
        let length : usize = X.len();
        for i in 0..length{
            res *= X[i];
        }
        res
    }
    fn search_vector(&self,X:&Vec<f64>,x:f64) -> f64{
        let denominator : usize = X.len();
        let mut numerator : f64 = 0.0;
        for i in 0..denominator{
            if X[i] == x{
                numerator += 1.0;
            }
        }
        let frac : f64 = numerator / denominator as f64;
        frac
    }
    fn walk_contin(&self,X:Vec<Vec<f64>> , vector: &Vec<f64> , prob : f64) -> f64{
        // we have to transpose each vector so that we can match each one
        let mut pdf_cache : Vec<f64> = Vec::new();
        let mut T : Vec<Vec<f64>> = Vec::new();
        if let Some(row_len) = X.first().map(|row| row.len()){
            T = (0..row_len).map(|i| X.iter().map(|row| row[i]).collect()).collect();
        }
        for column in T{
            let vector_length : usize = vector.len() - 1;
            for i in 0..vector_length{
                let pdf_out : f64 = self.norm_pdf(&column,vector[i]);
                if pdf_out == 0.0{
                    continue; // to avoid smoothing we remove the 0 densities
                }
                pdf_cache.push(pdf_out);
            }
        }
        self.prod_vec(pdf_cache) // this is the output after multiplying the class values f64 type
    }
    fn walk_cat(&self,X:Vec<Vec<f64>>, vector : &Vec<f64> , prob : f64) -> f64{
        let mut pdf_cache : Vec<f64> = Vec::new();
        pdf_cache.push(prob);
        let mut T : Vec<Vec<f64>> = Vec::new();
        if let Some(row_len) = X.first().map(|row| row.len()){
            T = (0..row_len).map(|i| X.iter().map(|row| row[i]).collect()).collect();
        }
        for column in T{
            let vector_length : usize = vector.len() - 1;
            for i in 0..vector_length{
                let pdf_out : f64 = self.search_vector(&column,vector[i]);
                if pdf_out == 0.0{
                    continue; // to avoid smoothing we remove the 0 densities
                }
                pdf_cache.push(pdf_out);
            }
        }
        self.prod_vec(pdf_cache) // this is the output after multiplying the class values f64 type
    }
    fn fit(&self, X: Vec<Vec<f64>>, new_record: Vec<f64>, continuous: bool) -> i32{
        if continuous == true{
            // split between either class
            let p : String = String::new();
            let classes = self.split_class(X);
            let true_class  = classes.1;
            let false_class = classes.0;
            let true_length : f64 = true_class.len() as f64;
            let false_length : f64 = false_class.len() as f64;
            let x_length : f64 = false_length + true_length;
            let true_prob : f64 = true_length /  x_length;
            let false_prob : f64 = false_length / x_length;
            // we also want to add the probability of the class being true or false
            let true_value : f64 = self.walk_contin(true_class , &new_record,true_prob); 
            let false_value : f64 = self.walk_contin(false_class , &new_record,false_prob);
            // this is the decision we return
            if true_value >= false_value{
                1
            }else{
                0
            }
        }else{
            // split between either class
            let classes = self.split_class(X);
            let true_class  = classes.1;
            let false_class = classes.0;
            let true_length : f64 = true_class.len() as f64;
            let false_length : f64 = false_class.len() as f64;
            let x_length : f64 = false_length + true_length;
            let true_prob : f64 = true_length /  x_length;
            let false_prob : f64 = false_length / x_length;
            // we also want to add the probability of the class being true or false
            let true_value : f64 = self.walk_cat(true_class , &new_record,true_prob);
            let false_value : f64 = self.walk_cat(false_class , &new_record,false_prob);
            // this is the decision we return
            if true_value >= false_value{
                1
            }else{
                0
            }
        }
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
impl CancerCsv{
    fn make_vec(&self) -> Vec<f64>{
        vec![self.SMOKING,self.YELLOW_FINGERS,self.ANXIETY,self.PEER_PRESSURE,self.CHRONICDISEASE,self.FATIGUE,self.ALLERGY,self.WHEEZING,
             self.ALCOHOLCONSUMING,self.COUGHING,self.SHORTNESSOFBREATH,self.SWALLOWINGDIFFICULTY,self.CHESTPAIN,self.LUNG_CANCER]
    }
    // we have to manually crate a vec for the struct
    
    
    
    
}
// this means we will load from a string path
// data has been halved
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



fn main() {
    let model = NaiveBayes;
    let train_data = load_csv("src/lcdata.csv").expect("Error Reading");
    let test_data = load_csv("src/test_data.csv").expect("Error Reading");
    let test_data_length : usize = test_data.len();
    let mut predictions = Vec::new();
    let ins : usize = test_data[0].len();
    for i in 0..test_data_length {
        let r_record: Vec<f64> = test_data[i][0..ins - 1].to_vec();
        let prediction = model.fit(train_data.clone(), r_record, false);
        predictions.push(prediction.clone());
    }
    println!("Predctions,{:?}",predictions);
}
