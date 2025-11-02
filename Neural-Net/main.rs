
// vector of the number of hidden layers we want to utilize -> hidden_layers
// the amount of nodes per layer will be any usize value
use std::collections::HashSet;
use ordered_float::NotNan;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::error::Error;
struct NeuralNet{
    hidden_layers:usize,
    nodes:Vec<usize>
}

impl NeuralNet{
    // make the new
    fn new(hidden_layers: usize , nodes : Vec<usize>) -> Self{
        Self{
            hidden_layers,
            nodes,
        }
    }
    fn sse(&self,x:f64,y:f64) -> f64{
        (x - y).powi(2)
    }
    fn sigmoid(&self,x:f64) -> f64{
        1.0 / (1.0 + (-x).exp()) // dont hardcode e
    }
    fn sse_sigmoid_gradient(&self,ground_truth:f64,prediction:f64 , xi:f64) -> f64{
        let delta : f64 = (ground_truth - prediction) * prediction * (1.0 - prediction) * xi;
        delta
    }
    fn matrix_multiplication(&self,matrix:Vec<Vec<f64>>,vector:Vec<f64>) -> Result<Vec<f64>,String>{
        // vector : (dim 1 * m) -> u , weight_matrix : (dim m * n) -> A
        // Multiply[vector * A^T]  : (dim 1 * n) -> v
        let m : usize = matrix.len();
        let n : usize = matrix[0].len(); // dimensions of the weight matrix
        let h : usize = vector.len();  // dimensions of the input vector
        // 
        if h != m {
            return Err("invalid matrix sizes , must be of size (1, m) X (m ,n)".to_string());
        }
        let mut res_vector : Vec<f64> = Vec::new();
        for i in 0..n{
            let mut ui : f64 = 0.0;
            for j in 0..vector.len(){
                ui += (vector[j] * matrix[j][i]);
            }
            res_vector.push(self.sigmoid(ui)); // every single perceptron will have a sigmoid activation
        }
        Ok(res_vector)
    }
    fn get_shape(&self,input_vector:&Vec<f64>) -> Vec<usize>{
        vec![1,input_vector.len()]
    }
    fn make_matrix(&self,m_length:usize,n_length:usize,default_weight:f64) -> Vec<Vec<f64>>{
        let mut matrix : Vec<Vec<f64>> = Vec::new();
        for i in 0..m_length{
            matrix.push(vec![default_weight;n_length]); // makes a matrix of length m with each m[i].len() == n
        }
        matrix
    }
    fn make_input(&self,n_length:usize) -> Vec<f64>{
        vec![0.0;n_length]
    }
    fn create_network(&self,weight:f64,input_vector:Vec<f64>) -> Result<(Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>),String>{
        if self.hidden_layers != self.nodes.len(){
            return Err("this is not a valid network shape (layers must be equal to node length)".to_string());
        }
        let input_shape = self.get_shape(&input_vector);
        let mut input_layers : Vec<Vec<f64>> = Vec::new();
        let mut weight_matrices : Vec<Vec<Vec<f64>>> = Vec::new();
        for i in 0..self.nodes.len(){
            let input = self.make_input(input_shape[1]);
            input_layers.push(input.clone());
            let layer  = self.make_matrix(input_shape[1],self.nodes[i],weight);
            weight_matrices.push(layer.clone());
            let input : Vec<f64> = self.matrix_multiplication(layer,input)?; //shadow the value for the next iteration
            let input_shape = self.get_shape(&input); // call to get shape of L_n+1 
            // this will be used in the next state of the program
        }
        // since this is a regression NN we need to edit the last layer as a 1D output so we will add this last value
        // also since input_shape is still in scope outside the loop we don't need to clone
        // we only need one more matrix
        let layer  = self.make_matrix(input_shape[1],1,weight);
        let output : Vec<f64> = self.matrix_multiplication(layer.clone(),input_layers[input_layers.len() - 1].clone())?;
        weight_matrices.push(layer);
        input_layers.push(output);
        // now input layers consist of the input shape and all hidden layers and the output layer
        // weight_matrices includes all weight matrices in-between
        Ok((input_layers,weight_matrices))
    }
    /*
    // Stochastic Gradient Descent
    // this implementation will use Stochastic Gradient Descent , all activations will be sigmoid defined 
    // this means our batch size will automatically be 1
    // Sum of Squared Error Loss (y - y')^ 2
    // We use Gradient Descent with momentum 
    // Below rust pseudo code shows how we implemented the back propagation
    // the model itself will be a serialized Impl that takes vectors and performs matrix operations and returns a prediction
    // we call this prediction y'
    
        let learning_rate : f64 = 0.000005;
        let beta : f64 = 0.3;
        let mut velocity : f64 = 0.0;
        let net = self.create_network(0.5,input_vec.clone()).expect("Invalid Network Architecture");
        let input: Vec<Vec<f64>> = net.0;
        let ln : Vec<f64> = input_vec.clone();
        let mut matrices : Vec<Vec<Vec<f64>>> = net.1;
        let mut input_track : Vec<Vec<f64>> = Vec::new();
        if input.len() != matrices.len(){
            return Err("Cannot compute Network".to_string());
        }else{
            // full network pass
            for i in 0..matrices.len(){
                let ln= self.matrix_multiplication(matrices[i].clone(),ln.clone()).expect("Invalid Size");
                input_track.push(ln);
            }
            // full back propagation
            for sigma in (0..input_track.len()).rev(){
                for index in (0..input_track[sigma].len()).rev(){
                    let mut weight_column_index : usize = 0;
                    for weight_column in (0..matrices[sigma][index].len()).rev(){
                        let weight_gradient : f64 = self.sse_sigmoid_gradient(ground_truth,input_track[sigma][weight_column_index],matrices[sigma][index][weight_column]);
                        velocity = (beta * velocity) + (1.0 - beta) * weight_gradient;
                        // then update w[i] = w[i] - velocity * learning rate
                        matrices[sigma][index][weight_column] = matrices[sigma][weight_column_index][weight_column] - (velocity * learning_rate);
                        weight_column_index += 1; 
                        // this way each column will get the row wise output variable
                        // {w_00(gradient-delta)xn , w_01(gradient-delta)xn-1 , w_02(gradient-delta)xn-2 ...... w_0n(gradient-delta)x1}
                        // {w_10(gradient-delta)xn , w_11(gradient-delta)xn-1 , w_12(gradient-delta)xn-2 ...... w_1n(gradient-delta)x1}
                        // {w_n0(gradient-delta)xn , w-n1(gradient-delta)xn-1 , w_n2(gradient-delta)xn-2 ...... w_nn(gradient-delta)x1}
                    }
                }
            }
            Ok((input,matrices))
        }
        
    }
     */
    fn fit(&self,X:Vec<Vec<f64>>,Y:Vec<f64>,epochs : usize , weight_start : f64) -> Result<(Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>),String>{
        
        let learning_rate : f64 = 0.008;
        let beta : f64 = 0.8;
        let mut velocity : f64 = 0.0;
        let network_shape : Vec<f64> = X[0].clone();
        let net = self.create_network(weight_start,network_shape).expect("Invalid Network Architecture");
        let input: Vec<Vec<f64>> = net.0;
        let mut matrices : Vec<Vec<Vec<f64>>> = net.1;
        let mut input_track : Vec<Vec<f64>> = Vec::new();
        
        if input.len() != matrices.len(){
            return Err("Cannot compute Network".to_string());
        }
        for i in 0..epochs{
            for j in 0..X.len(){
                // full network pass
                for k in 0..matrices.len(){
                    let ln= self.matrix_multiplication(matrices[k].clone(),X[j].clone()).expect("Invalid Size");
                    input_track.push(ln);
                }
                
                let last_value = input_track[input_track.len()-1][0];
                let model_error = self.sse(last_value,Y[j]);
                println!(" Current Model Error {:?} ",model_error);
                
                // full back propagation
                for a in (0..input_track.len()).rev(){
                    for b in (0..input_track[a].len()).rev(){
                        for c in (0..matrices.len()).rev(){
                            for d in (0..matrices[c].len()).rev(){
                                for e in (0..matrices[c][d].len()).rev(){
                                    let weight_gradient : f64 = self.sse_sigmoid_gradient(Y[j],input_track[a][b],input[a-1][b]); // fully connected

                                    velocity = (beta * velocity) + (1.0 - beta) * weight_gradient;

                                    // then update w[i] = w[i] - velocity * learning rate

                                    matrices[c][d][e] = matrices[c][d][e] - (velocity * learning_rate); // inplace update
                                    
                                }
                            }
                        }
                    }
                }
            }
            println!(" Matrices {:?}",matrices); // displaying weight change after epoch of training data
            input_track.clear()
        }
        
        Ok((input,matrices))
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
fn sigmoid(x:f64) -> f64{
    1.0 / (1.0 + (-x).exp()) // dont hardcode e
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

fn sse(x:f64,y:f64) -> f64{
    (x - y).powi(2)
}

fn matrix_multiplication(matrix:Vec<Vec<f64>>,vector:Vec<f64>) -> Result<Vec<f64>,String>{
    // vector : (dim 1 * m) -> u , weight_matrix : (dim m * n) -> A
    // Multiply[vector * A^T]  : (dim 1 * n) -> v
    let m : usize = matrix.len();
    let n : usize = matrix[0].len(); // dimensions of the weight matrix
    let h : usize = vector.len();  // dimensions of the input vector
    // 
    if h != m {
        return Err("invalid matrix sizes , must be of size (1, m) X (m ,n)".to_string());
    }
    let mut res_vector : Vec<f64> = Vec::new();
    for i in 0..n{
        let mut ui : f64 = 0.0;
        for j in 0..vector.len(){
            ui += (vector[j] * matrix[j][i]);
        }
        res_vector.push(sigmoid(ui)); // every single perceptron will have a sigmoid activation
    }
    Ok(res_vector)
}

fn predict(X:Vec<Vec<f64>> ,matrices:Vec<Vec<Vec<f64>>>)->f64{
    let y_vals = target(&X);
    let mut start : usize = 0;
    let mut err : Vec<f64> = Vec::new();
    let mut final_matrix : Vec<f64> = Vec::new();
    for k in 0..matrices.len(){
        let ln= matrix_multiplication(matrices[k].clone(),X[start].clone()).expect("Invalid Size");
        final_matrix = ln;
    }
    let e : f64 = sse(final_matrix[0],y_vals[start]);
    start += 1;
    err.push(e);
    
    err.iter().sum()
}



fn main() {
    let train_vec = load_csv("src/train.csv").expect("Error Loading File");
    let test_vec = load_csv("src/test.csv").expect("Error Loading File");
    let net = NeuralNet::new(3,vec![10,10,10]);
    let lung_cancer_train : Vec<f64> = target(&train_vec);
    let trained_weights = net.fit(train_vec,lung_cancer_train,8,0.5);
    let matrix_weights = trained_weights.unwrap().1;
    let predictions = predict(test_vec,matrix_weights);
    println!(" Total Model Error {:?} ", predictions);
    
    

    
    
    
    
    
 
}
