# Advanced Recommender System

## Project Contributors
- **Mouad AIT HA**
- **Mohamed LAKBAKBI**

## Overview

This project aims to develop an advanced recommender system using various machine learning and deep learning techniques. Our focus is on building and deploying models that can accurately recommend books to users based on their preferences. The project involves data collection, preprocessing, model building, and deployment.

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Models](#models)
4. [Technologies Used](#technologies-used)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Work](#future-work)
9. [License](#license)

## Introduction

The objective of this project is to build an advanced recommender system that leverages both classical and modern machine learning techniques. We utilized data from Amazon books, which includes three datasets: users, books, and ratings. Our system incorporates various models, ranging from baseline algorithms to sophisticated neural networks.

## Data

The project utilizes three datasets from Amazon:
1. **Users Dataset**: Contains information about the users.
2. **Books Dataset**: Contains details about the books.
3. **Ratings Dataset**: Contains user ratings for different books.

## Models

We have implemented the following models:
1. **Baseline Model**: Provides a simple benchmark for recommendations.
2. **K-Nearest Neighbors (KNN)**: Collaborative filtering based on user-user or item-item similarity.
3. **Singular Value Decomposition (SVD)**: Matrix factorization technique for collaborative filtering.
4. **Non-Negative Matrix Factorization (NMF)**: Another matrix factorization technique.
5. **Autoencoders**: Neural network-based approach for collaborative filtering.

## Technologies Used

Our project leverages a wide array of technologies and tools, including but not limited to:
- **Programming Languages and Environments**:
  - **Python**: Main programming language.
  - **Jupyter Notebook**: For interactive development.
  
- **Data Manipulation and Analysis**:
  - **Pandas**: Data manipulation and analysis.
  - **Numpy**: Numerical computing.
  - **SQLAlchemy**: SQL toolkit and Object-Relational Mapping (ORM) library.

- **Machine Learning and Deep Learning**:
  - **Scikit-Learn**: Machine learning library.
  - **TensorFlow**: Deep learning framework.
  - **Keras**: High-level neural networks API.

- **Data Visualization**:
  - **Matplotlib**: Plotting library.
  - **Seaborn**: Statistical data visualization.
  - **Plotly**: Interactive graphing library.

- **Natural Language Processing**:
  - **NLTK (Natural Language Toolkit)**: Text processing.

- **Model Deployment**:
  - **Flask**: Web framework for deploying models.
  - **Docker**: Containerization platform.

- **Version Control and Collaboration**:
  - **Git**: Version control system.
  - **GitHub**: Repository hosting service.

- **Other Tools and Libraries**:
  - **SciPy**: Scientific computing library.
  - **LightFM**: Hybrid recommendation algorithm library.
  - **Surprise**: Library for building and analyzing recommender systems.
  - **Optuna**: Hyperparameter optimization framework.

## Setup and Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/Advanced_recommender-PFA.git
    cd Advanced_recommender-PFA
    ```

2. **Create and Activate a Virtual Environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing**:
    - Navigate to the `data_preprocessing` directory and run the preprocessing scripts to prepare the datasets.

2. **Model Training**:
    - Navigate to the `models` directory.
    - Run the Jupyter notebooks to train the various models.

3. **Deployment**:
    - The models are deployed as part of a web application.
    - Navigate to the `deployment` directory and follow the instructions to deploy the application.

## Results

The results of our models are documented in the `results` directory. We provide detailed analyses, including performance metrics and visualizations, to demonstrate the effectiveness of each model.

## Future Work

Future improvements to this project may include:
- Enhancing the neural network models with more complex architectures.
- Incorporating additional data sources to improve recommendation accuracy.
- Implementing real-time recommendation systems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank our mentors and peers for their guidance and support throughout this project.

