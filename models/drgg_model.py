# models/sgg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align

class FreqBias(nn.Module):
    """
    Calcule un logit basé sur une matrice de fréquence des relations
    Paramètres:
      num_obj (int): Nombre de classes d'objets
      num_rel (int): Nombre de classes de relations
    """
    def __init__(self, num_obj, num_rel):
        super().__init__()
        self.num_obj = num_obj
        self.num_rel = num_rel
        # Matrice de fréquence de forme [num_obj, num_obj, num_rel]
        self.freq_matrix = nn.Parameter(torch.zeros(num_obj, num_obj, num_rel), requires_grad=False)

    def forward(self, pairs):
        """
        Calcule les logits de fréquence pour chaque paire
        Paramètres:
          pairs (torch.Tensor): Tenseur de forme [N_rel, 2] contenant (sujet, objet)
        Retourne:
          torch.Tensor: Tenseur de forme [N_rel, num_rel] des logits de fréquence
        """
        sc = pairs[:, 0]
        oc = pairs[:, 1]
        return self.freq_matrix[sc, oc, :]

    def index_with_labels(self, pairs):
        """
        Alias de forward
        Paramètres:
          pairs (torch.Tensor): Tenseur de forme [N_rel, 2]
        Retourne:
          torch.Tensor: Tenseur de forme [N_rel, num_rel]
        """
        return self.forward(pairs)

class DRGGRelationHead(nn.Module):
    """
    Tête de relations du modèle DRGG qui utilise un encodeur et un décodeur Transformer
    Paramètres:
      d_model (int): Dimension du modèle
      num_obj_classes (int): Nombre de classes d'objets
      num_rel_classes (int): Nombre de classes de relations
      num_layers (int): Nombre de couches dans l'encodeur et le décodeur
      num_queries (int): Nombre de requêtes pour le décodeur
    """
    def __init__(self, d_model=256, num_obj_classes=151, num_rel_classes=50,
                 num_layers=3, num_queries=100):
        super().__init__()
        self.d_model = d_model
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.num_layers = num_layers
        self.num_queries = num_queries

        # Création des couches d'encodeur Transformer
        encs = [nn.TransformerEncoderLayer(d_model, 8, 1024) for _ in range(num_layers)]
        # Création des couches de décodeur Transformer
        decs = [nn.TransformerDecoderLayer(d_model, 8, 1024) for _ in range(num_layers)]
        self.encoder_layers = nn.ModuleList(encs)  # Liste des couches d'encodeur
        self.decoder_layers = nn.ModuleList(decs)    # Liste des couches de décodeur

        # Embedding pour les requêtes du décodeur
        self.query_embed = nn.Embedding(num_queries, d_model)
        # Paramètres de pondération pour combiner les sorties des couches
        self.alpha_logit_enc = nn.Parameter(torch.zeros(num_layers))
        self.alpha_logit_dec = nn.Parameter(torch.zeros(num_layers))
        # MLP pour raffiner les features avec skip connections
        self.refine_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),  # Couche linéaire
            nn.ReLU(),                    # Activation ReLU
            nn.Linear(d_model, d_model)   # Couche linéaire
        )
        # Classifieur pour les objets
        self.obj_classifier = nn.Linear(d_model, num_obj_classes)
        # Tête pour prédire l'existence d'une relation
        self.rel_exist_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        # Tête pour prédire le type de relation
        self.rel_type_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_rel_classes)
        )

    def forward(self, node_feat, rel_inds):
        """
        Exécute la tête de relations sur les features des objets
        Paramètres:
          node_feat (torch.Tensor): Features des objets de forme [N_obj, d_model]
          rel_inds (torch.Tensor): Tenseur de forme [N_rel, 3] contenant (img_idx, index_sujet, index_objet)
        Retourne:
          tuple: (obj_dists, rel_dists) où obj_dists est de forme [N_obj, num_obj_classes] et rel_dists est de forme [N_rel, num_rel_classes]
        """
        device = node_feat.device
        N_obj = node_feat.size(0)
        # Préparation de l'encodeur (ajout d'une dimension pour le temps)
        x_enc = node_feat.unsqueeze(1)  # [N_obj, 1, d_model]
        enc_outs = []
        for layer in self.encoder_layers:
            x_enc = layer(x_enc)  # Passage par une couche d'encodeur
            # Raffinement avec MLP et ajout de la connexion résiduelle
            x_ref = self.refine_mlp(x_enc) + x_enc
            enc_outs.append(x_ref)
        memory = enc_outs[-1]
        # Préparation du décodeur
        query_pos = self.query_embed.weight.unsqueeze(1)
        tgt_init = torch.zeros_like(query_pos)
        dec_outs = []
        x_dec = tgt_init
        for layer in self.decoder_layers:
            x_dec = layer(x_dec, memory)
            x_ref = self.refine_mlp(x_dec) + x_dec
            dec_outs.append(x_ref)
        # Agrégation pondérée des sorties de l'encodeur
        enc_stack = torch.stack(enc_outs, dim=0)  # [num_layers, 1, N_obj, d_model]
        alpha_enc = F.softmax(self.alpha_logit_enc, dim=0).view(self.num_layers, 1, 1, 1)
        enc_agg = (enc_stack * alpha_enc).sum(dim=0).squeeze(0)  # [N_obj, d_model]
        # Agrégation pondérée des sorties du décodeur
        dec_stack = torch.stack(dec_outs, dim=0)
        alpha_dec = F.softmax(self.alpha_logit_dec, dim=0).view(self.num_layers, 1, 1, 1)
        dec_agg = (dec_stack * alpha_dec).sum(dim=0).squeeze(1)  # [num_queries, d_model]
        # Classification des objets
        obj_dists = self.obj_classifier(enc_agg)
        N_rel = rel_inds.size(0)
        if N_rel == 0:
            rel_dists = obj_dists.new_zeros((0, self.num_rel_classes))
        else:
            # Extraction des features pour le sujet et l'objet de chaque relation
            subj_enc = enc_agg[rel_inds[:, 1]]
            obj_enc = enc_agg[rel_inds[:, 2]]
            pair_enc = torch.cat([subj_enc, obj_enc], dim=-1)
            A_pred = self.rel_exist_head(pair_enc)
            G_pred = self.rel_type_head(pair_enc)
            A_sig = torch.sigmoid(A_pred)
            rel_dists = G_pred * A_sig
        return obj_dists, rel_dists

class MyRelModelDRGG(nn.Module):
    """
    Modèle complet de génération de graphes de scène (DRGG)
    Paramètres:
      num_obj_classes (int): Nombre de classes d'objets
      num_rel_classes (int): Nombre de classes de relations
      hidden_dim (int): Dimension cachée du modèle
      mode (str): Mode d'utilisation (ex: 'sgdet')
      use_bias (bool): Indique si le biais de fréquence est utilisé
      test_bias (bool): Indique si le biais de fréquence est appliqué en test
      num_layers (int): Nombre de couches dans la tête DRGG
      roi_size (int): Taille du ROI pour le ROI Align
    """
    def __init__(self, num_obj_classes=151, num_rel_classes=50, hidden_dim=256,
                 mode='sgdet', use_bias=True, test_bias=False, num_layers=3, roi_size=7):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.use_bias = use_bias
        self.test_bias = test_bias
        self.num_layers = num_layers
        self.roi_size = roi_size

        # Détecteur Faster R-CNN pré-entraîné (backbone + RPN + ROI heads)
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.eval()
        # Couche de projection pour convertir les features de ROI (256 * roi_size^2) en features de dimension cachée
        self.node_proj = nn.Linear(256 * (roi_size ** 2), hidden_dim)
        nn.init.xavier_uniform_(self.node_proj.weight)
        # Tête de relations DRGG
        self.relation_head = DRGGRelationHead(
            d_model=hidden_dim,
            num_obj_classes=num_obj_classes,
            num_rel_classes=num_rel_classes,
            num_layers=num_layers
        )
        # Biais de fréquence pour ajuster les logits de relation
        self.freq_bias = FreqBias(num_obj_classes, num_rel_classes)

    def train(self, mode=True):
        """
        Met le modèle en mode entraînement tout en gardant le détecteur en mode évaluation
        Paramètres:
          mode (bool): True pour entraînement, False pour évaluation
        Retourne:
          self
        """
        super().train(mode)
        self.detector.eval()
        return self

    def _dummy_detections(self, B, device):
        """
        Génère des détections vides pour B images
        Paramètres:
          B (int): Nombre d'images
          device (torch.device): Le device sur lequel créer les tenseurs
        Retourne:
          list: Liste de dictionnaires de détections vides pour chaque image
        """
        dummy = {
            'boxes': torch.empty((0, 4), device=device),
            'scores': torch.empty((0,), device=device),
            'labels': torch.empty((0,), device=device, dtype=torch.long)
        }
        return [dummy for _ in range(B)]

    @torch.no_grad()
    def faster_rcnn_forward(self, images, boxes, labels):
        """
        Exécute le détecteur Faster R-CNN sur les images pour obtenir les détections et les features
        Paramètres:
          images (list): Liste d'images (tenseurs) sur le device
          boxes (list): Liste de tenseurs de boîtes (sur CPU)
          labels (list): Liste de labels d'objets (sur CPU)
        Retourne:
          tuple: (detections, feat, input_sizes) où detections est la liste des détections, feat est le tenseur des features, et input_sizes est une liste des tailles d'images
        """
        self.detector.eval()
        detections = self.detector(images)
        transform_module = self.detector.transform
        image_list, _ = transform_module(images)
        backbone_out = self.detector.backbone(image_list.tensors)
        feat = list(backbone_out.values())[0]
        input_sizes = image_list.image_sizes
        return detections, feat, input_sizes

    def roi_align(self, feat, boxes, image_size):
        """
        Applique le ROI Align sur la feature map pour extraire les features des boîtes
        Paramètres:
          feat (torch.Tensor): La feature map de forme [B, 256, Hf, Wf]
          boxes (list): Liste de tenseurs de boîtes pour chaque image
          image_size (list): Liste des tailles d'images
        Retourne:
          tuple: (splitted, counts) où splitted est une liste de tenseurs pour chaque image et counts est une liste des nombres de boîtes par image
        """
        B = len(boxes)
        rois_list = []
        for i in range(B):
            b = boxes[i]
            N_i = b.size(0)
            if N_i == 0:
                continue
            idx_col = b.new_ones((N_i, 1)) * i
            rois_i = torch.cat((idx_col, b), dim=1)
            rois_list.append(rois_i)
        if len(rois_list) == 0:
            return None, []
        rois_full = torch.cat(rois_list, dim=0)
        Hf, Wf = feat.shape[2], feat.shape[3]
        Hi, Wi = image_size[0]
        scale_h = float(Hf) / Hi
        scale_w = float(Wf) / Wi
        spatial_scale = 0.5 * (scale_h + scale_w)
        pooled = roi_align(input=feat, boxes=rois_full, output_size=(self.roi_size, self.roi_size),
                           spatial_scale=spatial_scale)
        counts = [b.size(0) for b in boxes]
        splitted = []
        start = 0
        for c in counts:
            splitted.append(pooled[start:start + c])
            start += c
        return splitted, counts

    def build_rel_pairs(self, boxes):
        """
        Construit un ensemble de paires pour chaque image
        Paramètres:
          boxes (list): Liste de tenseurs de boîtes pour chaque image
        Retourne:
          torch.Tensor: Tenseur de forme [sumR, 3] contenant (img_idx, index_sujet, index_objet)
        """
        rel_inds_list = []
        B = len(boxes)
        for i in range(B):
            N_i = boxes[i].size(0)
            if N_i <= 1:
                continue
            pairs = []
            for s in range(N_i):
                for o in range(N_i):
                    if s != o:
                        pairs.append([i, s, o])
            pairs = torch.tensor(pairs, dtype=torch.long)
            rel_inds_list.append(pairs)
        if len(rel_inds_list) == 0:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.cat(rel_inds_list, dim=0)

    def forward(self, batch):
        """
        Exécute le modèle complet sur un batch d'images
        Paramètres:
          batch (dict): Dictionnaire contenant au moins les clés 'images', 'boxes', 'labels_obj'
        Retourne:
          dict: Dictionnaire contenant les sorties du modèle, y compris 'obj_logits_list', 'rel_logits_list', 'boxes', 'obj_scores', 'obj_preds', 'rel_pairs', 'rel_scores', et 'other_data'
        """
        images = batch['images']
        device = images[0].device
        with torch.no_grad():
            detections, feat, input_sizes = self.faster_rcnn_forward(
                images, batch['boxes'], batch['labels_obj']
            )
        B = len(images)
        if len(detections) != B:
            print(f"Warning: detections length ({len(detections)}) != number of images ({B}) - Filling missing detections with empty results")
            detections = self._dummy_detections(B, images[0].device)
        boxes_pred = []
        labels_pred = []
        for i in range(B):
            dt_i = detections[i]
            if dt_i['scores'].numel() > 0:
                keep_i = dt_i['scores'] > 0.3
            else:
                keep_i = torch.tensor([], dtype=torch.bool, device=dt_i['boxes'].device)
            boxes_i = dt_i['boxes'][keep_i]
            labels_i = dt_i['labels'][keep_i]
            boxes_pred.append(boxes_i)
            labels_pred.append(labels_i)
        splitted_pooled, counts = self.roi_align(feat, boxes_pred, input_sizes)
        if sum(counts) == 0 or splitted_pooled is None:
            return {
                'boxes': boxes_pred,
                'obj_scores': [torch.empty((0,), device=device) for _ in range(B)],
                'obj_preds': [torch.empty((0,), device=device, dtype=torch.long) for _ in range(B)],
                'rel_pairs': [torch.empty((0, 2), device=device, dtype=torch.long) for _ in range(B)],
                'rel_scores': [torch.empty((0, self.num_rel_classes), device=device) for _ in range(B)],
                'obj_logits_list': [torch.empty((0, self.num_obj_classes), device=device) for _ in range(B)],
                'rel_logits_list': [torch.empty((0, self.num_rel_classes), device=device) for _ in range(B)]
            }
        node_feats = []
        for sp in splitted_pooled:
            if sp.numel() == 0:
                nf = sp.new_empty((0, 256 * (self.roi_size ** 2)))
            else:
                nf = sp.view(sp.size(0), -1)
            node_feats.append(nf)
        node_feats = torch.cat(node_feats, dim=0)
        node_feats = self.node_proj(node_feats)
        rel_inds = self.build_rel_pairs(boxes_pred).to(device)
        offset = 0
        obj_offset_list = []
        for c in counts:
            obj_offset_list.append(offset)
            offset += c
        rel_inds_global = []
        for row in rel_inds:
            img_i, s_i, o_i = row.tolist()
            off = obj_offset_list[img_i]
            rel_inds_global.append([0, off + s_i, off + o_i])
        rel_inds_global = torch.tensor(rel_inds_global, dtype=torch.long, device=device)
        obj_dists, rel_dists = self.relation_head(node_feats, rel_inds_global)
        obj_logits = obj_dists
        rel_logits = rel_dists
        with torch.no_grad():
            sumN = obj_dists.size(0)
            if sumN > 0:
                probs_obj = F.softmax(obj_dists, dim=1)
                obj_preds = probs_obj.argmax(dim=1)
                obj_scores = probs_obj.max(dim=1)[0]
            else:
                obj_preds = torch.empty((0,), device=device, dtype=torch.long)
                obj_scores = torch.empty((0,), device=device)
            if rel_dists.size(0) > 0 and self.use_bias:
                subj_cls = obj_preds[rel_inds_global[:, 1]]
                obj_cls = obj_preds[rel_inds_global[:, 2]]
                pairs = torch.stack([subj_cls, obj_cls], dim=1)
                freq_log = self.freq_bias.index_with_labels(pairs)
                if self.test_bias:
                    rel_dists = freq_log
                else:
                    rel_dists = rel_dists + freq_log
            rel_probs = F.softmax(rel_dists, dim=1)
        out = {
            'boxes': boxes_pred,
            'obj_scores': [],
            'obj_preds': [],
            'rel_pairs': [],
            'rel_scores': [],
        }
        index_start = 0
        for i, c in enumerate(counts):
            if c == 0:
                out['obj_scores'].append(torch.empty((0,), device=device))
                out['obj_preds'].append(torch.empty((0,), device=device, dtype=torch.long))
                out['rel_pairs'].append(torch.empty((0, 2), device=device, dtype=torch.long))
                out['rel_scores'].append(torch.empty((0, self.num_rel_classes), device=device))
                continue
            sc_i = obj_scores[index_start:index_start + c]
            pr_i = obj_preds[index_start:index_start + c]
            out['obj_scores'].append(sc_i)
            out['obj_preds'].append(pr_i)
            relevant_ids = []
            for rid, row in enumerate(rel_inds_global):
                if row[1] >= index_start and row[1] < index_start + c:
                    relevant_ids.append(rid)
            relevant_ids = torch.tensor(relevant_ids, device=device)
            if relevant_ids.size(0) == 0:
                out['rel_pairs'].append(torch.empty((0, 2), device=device, dtype=torch.long))
                out['rel_scores'].append(torch.empty((0, self.num_rel_classes), device=device))
            else:
                pairs_loc = []
                sc_loc = []
                for rid in relevant_ids:
                    s_g = rel_inds_global[rid, 1].item()
                    o_g = rel_inds_global[rid, 2].item()
                    s_loc = s_g - index_start
                    o_loc = o_g - index_start
                    pairs_loc.append((s_loc, o_loc))
                    sc_loc.append(rid.item())
                pairs_loc = torch.tensor(pairs_loc, dtype=torch.long, device=device)
                sc_loc = torch.tensor(sc_loc, dtype=torch.long, device=device)
                out['rel_pairs'].append(pairs_loc)
                out['rel_scores'].append(rel_probs[sc_loc, :])
            index_start += c
        obj_logits_list = []
        start = 0
        for count in counts:
            end = start + count
            if count > 0:
                obj_logits_list.append(obj_logits[start:end])
            else:
                obj_logits_list.append(torch.empty((0, obj_logits.size(1)), device=obj_logits.device))
            start = end
        rel_counts = []
        for i in range(B):
            count_rel = sum(1 for row in rel_inds_global if row[1] >= obj_offset_list[i] and row[1] < (obj_offset_list[i] + counts[i]))
            rel_counts.append(count_rel)
        rel_logits_list = []
        start_rel = 0
        for count in rel_counts:
            end_rel = start_rel + count
            if count > 0:
                rel_logits_list.append(rel_logits[start_rel:end_rel])
            else:
                rel_logits_list.append(torch.empty((0, rel_logits.size(1)), device=rel_logits.device))
            start_rel = end_rel
        return {
            'obj_logits_list': obj_logits_list,
            'rel_logits_list': rel_logits_list,
            'boxes': boxes_pred,
            'obj_scores': out['obj_scores'],
            'obj_preds': out['obj_preds'],
            'rel_pairs': out['rel_pairs'],
            'rel_scores': out['rel_scores'],
            'other_data': batch.get('other_data', [])
        }

"""
Docstrings pour les sous-composantes de l'architecture

FreqBias:
  - Calcule un logit basé sur une matrice de fréquences des relations
  - La matrice a la forme [num_obj, num_obj, num_rel]

DRGGRelationHead:
  - Représente la tête de relations du modèle DRGG qui utilise un encodeur et un décodeur Transformer pour raffiner les features
  - La sortie est composée d'un tenseur pour la classification des objets et d'un tenseur pour les relations

MyRelModelDRGG:
  - Modèle complet de Scene Graph Generation combinant un détecteur Faster R-CNN et une tête DRGG
  - Utilise le détecteur pour obtenir des propositions d'objets et le ROI Align pour extraire les features
  - La tête DRGG produit les prédictions d'objets et de relations
"""
