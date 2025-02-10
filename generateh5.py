import json
import h5py
import numpy as np
from tqdm import tqdm
import os

def generate_vg_sgg_h5(image_data_file, objects_file, scene_graphs_file, output_file, image_root, max_data=20000):
    """
    Génère un fichier HDF5 contenant les informations Visual Genome
    (filenames, boxes, labels, relationships, etc.) en limitant à max_data images.

    :param image_data_file: Chemin vers image_data.json
    :param objects_file: Chemin vers objects.json
    :param scene_graphs_file: Chemin vers scene_graphs.json
    :param output_file: Chemin du fichier .h5 de sortie
    :param image_root: Dossier où sont stockées les images
    :param max_data: Nombre maximum d'images à traiter
    """

    # Charger les données JSON
    with open(image_data_file, 'r') as f:
        image_data = json.load(f)

    with open(objects_file, 'r') as f:
        objects_data = json.load(f)

    with open(scene_graphs_file, 'r') as f:
        scene_graphs_data = json.load(f)

    # Listes pour stocker les informations
    filenames = []
    bounding_boxes = []
    labels = []
    relationships = []
    
    # Indices reliant chaque box à son image
    box_to_img = []
    # Indices reliant chaque relation à son image
    rel_to_img = []

    data_count = 0  # Compteur pour le nombre d'images

    # Parcourir les images
    for img_data in tqdm(image_data, desc="Processing images"):
        if data_count >= max_data:
            break

        img_id = img_data['id']
        filename = os.path.join(image_root, f"{img_id}.jpg")
        
        # Récupérer les objets de cette image
        objects = next((item['objects'] for item in objects_data if item['id'] == img_id), None)
        if objects is None:
            continue
        
        # Mémoriser le chemin de l'image
        filenames.append(filename)

        # Ajouter toutes les bounding boxes + labels
        for obj in objects:
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            bounding_boxes.append(bbox)
            labels.append(obj['names'][0].encode('utf-8'))  # encode le texte
            box_to_img.append(data_count)

        # Récupérer les relations (scene_graphs.json)
        scene = next((scene for scene in scene_graphs_data if scene['image_id'] == img_id), None)
        if scene is not None:
            for rel in scene.get('relationships', []):
                relationships.append((
                    rel['subject_id'],
                    rel['predicate'].encode('utf-8'),
                    rel['object_id']
                ))
                rel_to_img.append(data_count)

        data_count += 1

    # Conversion en arrays NumPy
    filenames = np.array([fn.encode('utf-8') for fn in filenames], dtype='S')  # M
    bounding_boxes = np.array(bounding_boxes, dtype=np.int32)  # (N, 4)
    labels = np.array(labels, dtype='S')  # (N,)
    box_to_img = np.array(box_to_img, dtype=np.int32)  # (N,)
    rel_to_img = np.array(rel_to_img, dtype=np.int32)  # (R,)

    # Relations sous forme de structured array
    relationships_dtype = np.dtype([
        ('subject_id', 'i4'),
        ('predicate', 'S50'),
        ('object_id', 'i4')
    ])
    relationships = np.array(relationships, dtype=relationships_dtype)  # (R,)

    # Sauvegarde dans un fichier HDF5
    with h5py.File(output_file, 'w') as h5f:
        # split (0 pour train, par ex.)
        h5f.create_dataset('split', data=np.zeros(data_count, dtype=np.int32))
        # noms de fichiers
        h5f.create_dataset('filenames', data=filenames)
        # bounding boxes
        h5f.create_dataset('boxes_1024', data=bounding_boxes)
        h5f.create_dataset('labels', data=labels)
        h5f.create_dataset('box_to_img', data=box_to_img)
        # relations
        h5f.create_dataset('relationships', data=relationships)
        h5f.create_dataset('rel_to_img', data=rel_to_img)

    print(f"[OK] Fichier HDF5 généré : {output_file}")
    print(f"   {data_count} images, {len(bounding_boxes)} boxes, {len(relationships)} relations.")


if __name__ == "__main__":
    # Exemple d'utilisation
    image_data_file = "./data/image_data.json"
    objects_file = "./data/objects.json"
    scene_graphs_file = "./data/relationships.json"
    output_file = "./data/VG-SGG.h5"
    image_root = "./data/VG_100K"
    max_data = 10000

    generate_vg_sgg_h5(
        image_data_file, 
        objects_file, 
        scene_graphs_file, 
        output_file,
        image_root,
        max_data=max_data
    )
