# RQ-VAE Recommender
This project is forked from EdoardoBotta/RQ-VAE-Recommender, a PyTorch implementation of a generative retrieval model using semantic IDs based on RQ-VAE from "Recommender Systems with Generative Retrieval". 
The model has two stages:
1. Items in the corpus are mapped to a tuple of semantic IDs by training an RQ-VAE (figure below).
2. Sequences of semantic IDs are tokenized by using a frozen RQ-VAE and a transformer-based is trained on sequences of semantic IDs to generate the next ids in the sequence.
![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5).

### Currently supports
* **Datasets:** MovieLens 1M
* RQ-VAE Pytorch model implementation + KMeans initialization + RQ-VAE training script.
* RQ-VAE pre-trained checkpoints within 500,000 iterations + RQ-VAE inference script + output csv file.
* Decoder-only retrieval model + Training code with semantic id user sequences from randomly initialized RQ-VAE.

### Executing
* **Conda environment configuration:** 
`conda create -n rqvae_env python=3.9 -y`
`conda activate rqvae_env`
`pip install -r requirements.txt`
* **RQ-VAE tokenizer model training:** 
Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py`
* **RQ-VAE tokenizer model inference:** 
Trains the RQ-VAE tokenizer on the item corpus. Executed via `python inference_rqvae.py`
* **Retrieval model training:** 
Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py`

### Next steps
* Initalize RQ-VAE from pre-trained checkpoint + ML1M timestamp-based train/test split.
* Comparison encoder-decoder model vs. decoder-only model.

### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
* [Categorical Reparametrization with Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) by Eric Jang, Shixiang Gu, Ben Poole
  
