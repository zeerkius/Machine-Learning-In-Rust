
use std::collections::HashSet;
use ordered_float::NotNan;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::error::Error;
struct Data{
    XTest : Vec<Vec<f64>>,
    YTest : Vec<Vec<f64>>,
    XTrain : Vec<Vec<f64>>,
    YTrain : Vec<Vec<f64>>
}

struct LogisticRegression;

impl LogisticRegression{
    fn dot_product(&self,u:&Vec<f64>,v:&Vec<f64>) -> Result<f64,String>{
        let lengthu : usize  = u.len();
        let lengthv : usize = v.len();
        let mut res : f64 = 0.0;
        if lengthu != lengthv{
            return Err("Vectors must be same size".to_string());
        }else{
            for i in 0..lengthu{
                res += (u[i] * v[i]);
            }
        }
        Ok(res)
    }
    fn sigmoid(&self,x:f64) -> f64{
        let e : f64 = 2.7182818;
        let numerator : f64 = 1.0;
        let denominator : f64 = (1.0 + e.powf(-x));
        let sig : f64 = numerator / denominator;
        sig
    }
    fn sigmoid_gradient(&self,prediction:f64,ground_truth:f64) -> f64{
        let delta : f64 = (prediction - ground_truth) * prediction * (1.0 - prediction);
        delta
    }
    fn sse(&self,guess:f64 , prediction:f64) -> f64{
        let error = (guess - prediction).powi(2);
        error
    }
    fn fit(&self, X:&Vec<Vec<f64>>, Y: Vec<f64> , batch_size : i32 , epochs :i32) -> Result<Vec<f64>,String>{
        let beta : f64 = 0.3;
        let data_length : usize = X.len();
        let input_vector_length : usize = X[0].len();
        let learning_rate : f64 = 0.0000005;
        let mut weight_vector = vec![0.1;input_vector_length];
        let mut velocity : f64 = 0.0;
        let mut error_cache : Vec<Vec<f64>> = (0..input_vector_length).map(|_|  Vec::new()).collect();
        let mut batch_counter : i32 = 0;
        if batch_size <= 0{
            return Err("Batch Size must be greater than 0".to_string());
        }
        if epochs <= 0{
            return Err("Epochs must be greater than 0".to_string());
        }
        for i in 0..epochs{
            for j in 0..data_length{
                batch_counter += 1;
                if batch_size % batch_counter == 0{
                    // update weights
                    println!("Current Weights,{:?}",weight_vector);
                    for k in 0..input_vector_length{
                        let mut sum : f64 = error_cache[k].iter().sum();
                        velocity = (beta * velocity) +  (1.0 - beta) * sum;
                        weight_vector[k] -= velocity * learning_rate;
                    }
                }else{
                    let mut dot_prod : f64 = self.dot_product(&weight_vector,&X[j]).unwrap();
                    let mut sig_pred : f64 = self.sigmoid(dot_prod);
                    for k in 0..input_vector_length{
                        let mut graident : f64 = self.sigmoid_gradient(sig_pred,Y[j]);
                        error_cache[k].push(graident);
                    }
                }
            }
        }
        Ok(weight_vector)

    }
    fn predict(&self,X:Vec<Vec<f64>>,Y:Vec<f64> , W:&Vec<f64>)-> Vec<f64>{
        let mut predictions : Vec<f64> = Vec::new();
        let test_vector_size : usize = X.len();
        for i in 0..test_vector_size{
            let mut guess : f64 = self.dot_product(&W,&X[i]).unwrap();
            let mut guess : f64 = self.sigmoid(guess);
            predictions.push(guess);
        }
        let pred_sum : f64 = predictions.iter().sum();
        let ground_truth_sum : f64 = Y.iter().sum();
        let err = self.sse(pred_sum , ground_truth_sum);
        println!(" This is the Sum of Squared Error {}",err);
        predictions
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

fn target(X:&Vec<Vec<f64>>) -> Vec<f64>{
    let row_len : usize = X[0].len();
    // transpose to get the vectors column wise
    let mut T : Vec<Vec<f64>> = Vec::new();
    if let Some(row_len) = X.first().map(|row| row.len()){
        T = (0..row_len).map(|i| X.iter().map(|row| row[i]).collect()).collect();
    }
    let res : Vec<f64> = T[row_len - 1].to_vec();
    res
}


fn main() {
    // Needs testing with loading data
    let model = LogisticRegression;
    let train_vec = load_csv("src/train.csv").expect("Error Loading File");
    let test_vec = load_csv("src/test.csv").expect("Error Loading File");
    let lung_cancer_train : Vec<f64> = target(&train_vec);
    let weights = model.fit(&train_vec,lung_cancer_train,150,33).unwrap();
    let lung_cancer_test : Vec<f64> = target(&test_vec);
    let res : Vec<f64> = model.predict(test_vec,lung_cancer_test,&weights);
}


