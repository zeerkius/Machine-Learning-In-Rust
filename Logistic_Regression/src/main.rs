
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
    fn sigmoid_gradient(&self,prediction:f64,ground_truth:f64 , xi : f64) -> f64{
        let delta : f64 = (prediction - ground_truth) * prediction * (1.0 - prediction) * xi;
        delta
    }
    fn sse(&self,guess:f64 , prediction:f64) -> f64{
        let error = (guess - prediction).powi(2);
        error
    }
    fn fit(&self, X:Vec<Vec<f64>>, Y: Vec<f64> , batch_size : i32 , epochs :i32) -> Result<Vec<f64>,String>{
        let beta : f64 = 0.0;
        let data_length : usize = X.len();
        let input_vector_length : usize = X[0].len();
        let learning_rate : f64 = 0.000005;
        let mut weight_vector = vec![0.5;input_vector_length];
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
                    for k in 0..input_vector_length{
                        let mut sum : f64 = error_cache[k].iter().sum();
                        velocity += (1.0 - beta) * sum * (1.0 / batch_size as f64);
                        weight_vector[k] -= velocity * learning_rate;
                    }
                }else{
                    let mut dot_prod : f64 = self.dot_product(&weight_vector,&X[j]).unwrap();
                    let mut sig_pred : f64 = self.sigmoid(dot_prod);
                    for k in 0..input_vector_length{
                        let mut graident : f64 = self.sigmoid_gradient(sig_pred,Y[j],X[j][k]);
                        error_cache[k].push(graident);
                    }
                }
            }
        }
        Ok(weight_vector)

    }
    fn predict(&self,X:Vec<Vec<f64>>,Y:Vec<f64> , W:Vec<f64>)-> Vec<f64>{
        let mut predictions : Vec<f64> = Vec::new();
        let test_vector_size : usize = X.len();
        for i in 0..test_vector_size{
            let mut guess : f64 = self.dot_product(&W,&X[i]).unwrap();
            predictions.push(guess);
        }
        let pred_sum : f64 = predictions.iter().sum();
        let ground_truth_sum : f64 = Y.iter().sum();
        let err = self.sse(pred_sum , ground_truth_sum);
        println!(" This is the Sum of Squared Error {}",err);
        predictions
    }
}

fn main() {
    println!("Hello, world!");
}
