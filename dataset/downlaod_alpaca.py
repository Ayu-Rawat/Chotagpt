from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train")
ds.to_json("dataset/alpaca.jsonl")