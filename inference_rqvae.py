import torch
from data.schemas import SeqBatch
from modules.rqvae import RqVae
from data.ml1m import RawMovieLens1M
from torch.utils.data import DataLoader
from data.movie_lens import MovieLensMovieData
import csv

dataset_path = "dataset/ml-1m"
checkpoint_path = "inference_checkpoints/inference_model_iter_500000.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1

def load_rqvae_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint["config"]
    model = RqVae(
        input_dim=config["vae_input_dim"],
        embed_dim=config["vae_embed_dim"],
        hidden_dims=[config["vae_hidden_dim"]],
        codebook_size=config["vae_codebook_size"],
        n_layers=config["vae_n_layers"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def get_dataloader(dataset_path):
    dataset = MovieLensMovieData(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def generate_semantic_ids(model, dataloader, device, gumbel_t=0.5):
    model.eval()
    all_sem_ids = []
    
    for batch in dataloader:
        batch = SeqBatch(
            user_ids=batch.user_ids.to(device),
            ids=batch.ids.to(device),
            x=batch.x.to(device),
            seq_mask=batch.seq_mask.to(device)
        )
        with torch.no_grad():
            output = model.get_semantic_ids(batch.x, gumbel_t=gumbel_t)
            sem_ids = output.sem_ids.cpu().numpy()
            all_sem_ids.append(sem_ids)
    return all_sem_ids


if __name__ == "__main__":
    dataloader = get_dataloader("dataset/ml-1m")
    model = load_rqvae_model(checkpoint_path, device)
    output_file = "semantic_ids_output.csv"
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_ids", "semantic_ids"])
        
        for batch in dataloader:
            batch = SeqBatch(
                user_ids=batch.user_ids.to(device),
                ids=batch.ids.to(device),
                x=batch.x.to(device),
                seq_mask=batch.seq_mask.to(device)
            )
            with torch.no_grad():
                output = model.get_semantic_ids(batch.x, gumbel_t=0.5)

            semantic_ids = output.sem_ids.cpu().numpy()
            movie_ids = batch.ids.cpu().numpy()

            for i in range(len(movie_ids)):
                movie_id_sequence = int(movie_ids[i])
                sem_id_sequence = list(semantic_ids[i]) 
                writer.writerow([movie_id_sequence, sem_id_sequence])
    
    print(f"Results saved to {output_file}")
