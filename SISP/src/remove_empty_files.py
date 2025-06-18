import os
import time
from tqdm import tqdm

def remove_empty_files(input_dir):
    files_removed_cnt = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            txt_path = os.path.join(input_dir, filename)
            with open(txt_path, "r") as file:
                content = file.read().strip()
            
            if not content:  # Se o arquivo estiver vazio
                base_name = os.path.splitext(filename)[0]
                time.sleep(0.1)  # Pequeno atraso para evitar bloqueios
                for file_to_remove in os.listdir(input_dir):
                    if file_to_remove.startswith(base_name):
                        file_path = os.path.join(input_dir, file_to_remove)
                        try:
                            os.remove(file_path)
                            files_removed_cnt += 1
                            print(f"Arquivo excluído: {file_to_remove}")
                        except PermissionError:
                            print(f"Erro ao excluir {file_to_remove}, possivelmente em uso.")
    print(f"Número de arquivos vazios: {files_removed_cnt}")

# Exemplo de uso
input_directory = "./train"
remove_empty_files(input_directory)