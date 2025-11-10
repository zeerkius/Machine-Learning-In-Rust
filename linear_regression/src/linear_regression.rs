// This is a rust implementation of a Linear Regression Model
// We need a struct to Define Training / Testing Data
use std::collections::HashSet;
use ordered_float::NotNan;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::error::Error;
struct Data{
    X_train : Vec<Vec<f64>>,
    Y_train : Vec<f64>,
    X_test : Vec<Vec<f64>>,
    Y_test : Vec<f64>,
}

struct LRModel; // Meaning it has a type of LRModel , similarly in python it will have a type of the class name
// The Model Will only support SSE
impl LRModel{
    fn dot_product(&self,u:&Vec<f64>,v:&Vec<f64>)->Result<f64,String>{
        // handle errors of invalid length
        if u.len() != v.len(){
            return Err("Invalid Vector Length".to_string());
        }else{
            let length : usize = u.len();
            let mut dot_prod : f64 = 0.0;
            for i in 0..length{
                dot_prod += (u[i] * v[i]);
            }
            Ok(dot_prod) // preferred return type
        }
    }
    fn sse_gradient(&self,prediction : f64 , ground_truth : f64,xi:f64) -> f64{
        // we purposely omit the constant
        let delta : f64 = (prediction - ground_truth) * xi;
        delta
    }
    
    fn sse(&self,prediction : f64 , ground_truth: f64)  -> f64{
        let err : f64 = (prediction - ground_truth).powi(2);
        err
    }
    
    fn fit(&self,X:&Vec<Vec<f64>>,Y:Vec<f64>, batch_size : i32 , epochs : i32) -> Result<Vec<f64>,String>{
        let length_input_vector : usize = X[0].len();
        let length_training_data : usize = X.len();
        let length_ground_truth: usize = Y.len();
        if length_ground_truth != length_training_data{
            return Err("Incompatible Data Size Not enough ground truth for given examples".to_string());
        }
        if batch_size <= 0{
            return Err("Batch size must be greater than 0".to_string());
        }
        if epochs <= 0{
            return Err("Epochs must be greater than 0".to_string());
        }
        let learning_rate : f64 = 0.0000005; // moving the delta
        let beta : f64 = 0.3; // momentum-optimized gradient
        let mut velocity : f64 = 0.0;
        let mut weight_vector = vec![0.0;length_input_vector]; // this needs to be dynamically created based on input shape , also initialized at 0.5
        let mut error_cache : Vec<Vec<f64>> = (0..length_input_vector).map(|_| Vec::new()).collect(); // making empty vector to record error
        let mut batch_counter :i32 = 0;
        // iterate through the dataset for the amount of epochs the user wants
        // when batch size accumulate update weights
        // otherwise use weights for predictions
        for i in 0..epochs{
            for j in 0..length_training_data{
                batch_counter += 1;
                if batch_counter % batch_size == 0{ // avoid manual expensive copying and counting
                    // weight update
                    println!("Current Weights {:?}",weight_vector);
                    for k in 0..length_input_vector{
                        let mut sum : f64 = error_cache[k].iter().sum(); // avoid type ref
                        velocity += (beta * velocity) +  (1.0 - beta) * sum;
                        weight_vector[k] -= velocity * learning_rate
                    }
                    let mut error_cache : Vec<Vec<i32>> = (0..length_input_vector).map(|_| Vec::new()).collect(); //shadows and avoids moving errors
                }else{
                    let mut yi : f64 = self.dot_product(&weight_vector,&X[j]).unwrap(); // point to values to avoid moving errors
                    for n in 0..length_input_vector{
                        error_cache[n].push(self.sse_gradient(yi,Y[j],X[j][n]));
                    }
                }
            }
        }
        Ok(weight_vector)
    }
    
    fn predict(&self,X:Vec<Vec<f64>>,Y:Vec<f64> , trained_weights : &Vec<f64>) -> Vec<f64>{
        let mut predictions : Vec<f64> = Vec::new();
        let mut test_data_length : usize = X.len();
        for m in 0..test_data_length{
            let arr : &Vec<f64> = &X[m]; // cast splice -> vec pointer to avoid errors with dot product function
            let mut p : f64 = self.dot_product(arr,trained_weights).unwrap();
            predictions.push(p);
        }
        let p_sum : f64 = predictions.iter().sum();
        let ground_truth_sum : f64 = Y.iter().sum();
        let total_error : f64 = self.sse(p_sum,ground_truth_sum);
        println!(" Model Sum Of Squared Error {:?} ",total_error);
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
    let model = LRModel;
    let train_vec = load_csv("src/train.csv").expect("Error Loading File");
    let test_vec = load_csv("src/test.csv").expect("Error Loading File");
    let lung_cancer_train : Vec<f64> = target(&train_vec);
    let weights = model.fit(&train_vec,lung_cancer_train,150,31).unwrap();
    let lung_cancer_test : Vec<f64> = target(&test_vec);
    let res : Vec<f64> = model.predict(test_vec,lung_cancer_test,&weights);
}
