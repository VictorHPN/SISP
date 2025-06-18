import os
import time
from tqdm import tqdm

def convert_to_yolo_format(input_dir):
    files_rewritten_cnt = 0
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    # for filename in os.listdir(input_dir):
    for filename in tqdm(txt_files, desc="Convertendo arquivos", unit="arquivo"):
        # if filename.endswith(".txt"):  # Process only text files
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(input_dir, filename)  # Overwriting the same file
        
        with open(input_path, "r") as file:
            lines = file.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 7:
                continue  # Skip invalid lines
            
            obj_class = parts[0]
            if obj_class == "2":
                obj_class = "1"  # Convert class 2 to class 1
            
            x_top_left = float(parts[1])
            y_top_left = float(parts[2])
            box_width  = float(parts[3])
            box_height = float(parts[4])
            img_width  = float(parts[5])
            img_height = float(parts[6])
            
            # Compute YOLO format values
            x_center = (x_top_left + (box_width / 2)) / img_width
            y_center = (y_top_left + (box_height / 2)) / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height
            
            new_line = f"{obj_class} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
            new_lines.append(new_line)
        
        with open(output_path, "w") as file:
            file.writelines(new_lines)
            
        if new_lines:
            files_rewritten_cnt += 1
        #     print(f"Arquivo reescrito: {filename}")
    
    print(f"Número de arquivos reescritos: {files_rewritten_cnt}")
    print("Conversao concluida!\n")

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
convert_to_yolo_format(input_directory)
remove_empty_files(input_directory)
