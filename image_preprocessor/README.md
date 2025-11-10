This project is a high-performance image pre-processing pipeline built in Rust, designed to streamline data preparation for machine learning workflows. It efficiently processes raw image datasets through a series of customizable pre-processing stages—such as resizing, normalization, and format conversion—and outputs clean, structured data in CSV format, ready for model training.

✨ Key Features

⚡ Blazing-fast performance powered by Rust’s zero-cost abstractions

🧠 ML-ready output — automatically generates a .csv file compatible with most ML frameworks

🧩 Modular pre-processing stages (e.g., resizing, normalization, grayscale conversion, and more)

🧰 Customizable pipeline — easily extend or modify steps for your specific dataset

📦 Lightweight and reliable with minimal dependencies

🚀 Ideal For

Preparing large image datasets for machine learning or deep learning tasks

Converting raw image folders into structured numerical representations

Researchers and developers who value speed, safety, and reproducibility

🧪 Example Workflow
# Run the pre-processor on a dataset directory
cargo run -- --input ./dataset --output ./processed/data.csv


This command:

Loads images from ./dataset

Applies your chosen pre-processing pipeline

Exports features and labels to data.csv

🛠️ Tech Stack

Language: Rust 🦀

Data Format: CSV for ML compatibility

Focus: Performance, safety, and modularity
