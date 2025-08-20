
struct NeuralNet{
    hidden_layers:usize,
    nodes:Vec<usize>
}

impl NeuralNet{
    // make the new
    fn new(&self,hidden_layers: usize , nodes : Vec<usize>) -> Self{
        Self{
            hidden_layers,
            nodes,
        }
    }
    fn sse(&self,x:f64,y:f64) -> f64{
        (x - y).powi(2)
    }
    fn sigmoid(&self,x:f64) -> f64{
        let e : f64 = 2.718281828;
        let numerator : f64 = 1.0;
        let denominator : f64 = (1.0 + e.powf(-x));
        let res : f64 = (numerator / denominator);
        res
    }
    fn sse_sigmoid_gradient(&self,ground_truth:f64,prediction:f64,xi:f64) -> f64{
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
            res_vector.push(ui);
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
    fn SGD(&self) -> Result<(Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>),String>{
        // first we have to feed foward in the network and change weights after a full network pass
        
        {}
        
    }

    

}

fn main() {
    {}
    
}
