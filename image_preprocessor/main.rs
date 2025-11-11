use image::DynamicImage;
use image::imageops::FilterType;
use std::error::Error;
use csv::WriterBuilder;

struct ProcessConv{
    folder_name:String
}

impl ProcessConv{
    fn new(folder_name:String) -> Self{
        Self{
            folder_name,
        }
    }
    fn dot_product(&self,arr1:Vec<f32>,arr2:Vec<f32>) -> Result<f32,String>{
        if arr1.len() != arr2.len(){
            return Err("Invalid patch size".to_string());
        }
        let mut product : f32 = 0.0;
        for i in 0..arr1.len(){
            product += (arr1[i] * arr2[i]);
        }
        Ok(product)
    }
    fn convolute_horizontal_edge(&self,image_vec:Vec<f32>) -> f32{
        let h_edge_matrix : Vec<f32> = vec![1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,1.0,1.0];
        self.dot_product(image_vec,h_edge_matrix).unwrap()

    }

    fn convolute_vertical_edge(&self,image_vec:Vec<f32>) -> f32{
        let v_edge_matrix : Vec<f32> = vec![-1.0,0.0,1.0,-1.0,0.0,1.0,-1.0,0.0,1.0];
        self.dot_product(image_vec,v_edge_matrix).unwrap()

    }

    fn convolute_sharpen(&self,image_vec:Vec<f32>) -> f32{
        let sharpen_matrix : Vec<f32> = vec![0.0,-1.0,0.0,-1.0,5.0,-1.0,0.0,-1.0,0.0];
        self.dot_product(image_vec,sharpen_matrix).unwrap()


    }

    fn convolute_box_blur(&self,image_vec:Vec<f32>) -> f32{
        let blur_matrix : Vec<f32> = vec![0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111];
        self.dot_product(image_vec,blur_matrix).unwrap()
    }
    // we can change this to see how behaviour changes
    fn l_conv(&self ,pixel_buffer : Vec<f32>) -> Vec<f32> {
        let mut left_pointer : usize = 0;
        let mut right_pointer : usize = 8;
        let mut res_vec : Vec<f32> = Vec::new();

        for i in 0..(pixel_buffer.len() / 9) { // -> 3 pixels -> 1 pixel
            let slice: Vec<f32> = pixel_buffer[left_pointer..right_pointer].to_vec();
            let cnv_pixel: f32 = self.convolute_horizontal_edge(slice);
            res_vec.push(cnv_pixel);
            left_pointer += 9;
            right_pointer += 9;
        }
        res_vec
    }
    fn r_conv(&self ,pixel_buffer : Vec<f32>) -> Vec<f32> {
        let mut left_pointer : usize = 0;
        let mut right_pointer : usize = 8;
        let mut res_vec : Vec<f32> = Vec::new();

        for i in 0..(pixel_buffer.len() / 9) { // -> 3 pixels -> 1 pixel
            let slice: Vec<f32> = pixel_buffer[left_pointer..right_pointer].to_vec();
            let cnv_pixel: f32 = self.convolute_vertical_edge(slice);
            res_vec.push(cnv_pixel);
            left_pointer += 9;
            right_pointer += 9;
        }
        res_vec
    }



    fn load_process_conv_save(&self , n : u32 , folder_name : String) -> Result<Vec<Vec<f32>>,String>{
        if n % 3 != 0{
            return Err("Invalid shape size dimension must be divisible by 3".to_string());
        }
        let mut resized_images : Vec<Vec<f32>> = Vec::new();
        // we need to make this a directory first
        let img_folder = std::path::Path::new(&folder_name);
        let imgs = std::path::Path::read_dir(img_folder).expect("Folder Failed");
        // we are expecting all the file types to be the same
        for im in imgs{
            let im = im.expect("Image cant be loaded");
            let im_path = im.path();

            // we would like img types to be the same however image supports plenty of img types so we should be good
            // we default all images to be n x n
            let image_vectors : DynamicImage = image::open(&im_path).unwrap();
            let compressed = image_vectors.resize_exact(n.clone(),n.clone(),FilterType::Nearest);

            let compressed  : Vec<f32> = compressed.to_rgb32f().to_vec(); // flattens the output with f32
            // as this is just simply a (n x n x 3) tensor I can just preprocess along the entire vector every 9 f32 values
            // since we know our dim is  72 x 72 we can do 2 convolutions and get an 8 x 8 x 3 dim img which we can process through our model
            // this makes it a 192 shape vector

            // convolution step

            let first_conv = self.l_conv(compressed);
            let second_conv = self.r_conv(first_conv);
            // currently only doing horizontal and vertical edges

            resized_images.push(second_conv);
            }
        Ok(resized_images)
        }
        fn make_csv(&self,X:Vec<Vec<f32>>){
            ()
            // save pre-proccessed data to a csv
            
            
        }














    }








fn main() {



    println!("Hello, world!");
}
