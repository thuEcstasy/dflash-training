import requests
import os
r = requests.get(                                                                                                               
    "https://hf-mirror.com/api/datasets/nvidia/Nemotron-Post-Training-Dataset-v1/parquet/chat",
    verify=False, timeout=30                                                                                                    
)                                                                                                                               
print(r.status_code)                                                                                                            
print(r.json()) 