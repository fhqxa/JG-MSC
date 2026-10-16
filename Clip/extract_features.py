"""
extract_features.py
提取CLIP图像特征和文本prompt特征，用于训练/推理
"""

import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip
from datasets_split.dtd import DescribableTextures
from datasets_split.utils import build_data_loader
import torchvision.transforms as transforms


def extract_text_feature_all(cache_dir, classnames, clip_model, template,
                             prompt_path=None, norm=True):
    """
    提取所有prompt的文本特征

    Args:
        cache_dir: 保存路径
        classnames: 类别名列表
        clip_model: CLIP模型
        template: prompt模板列表
        prompt_path: CuPL prompt文件路径（可选）
        norm: 是否归一化
    """
    print("\n" + "=" * 60)
    print("Extracting ALL text prompt features...")
    print("=" * 60)

    # 加载额外的prompts（如果有CuPL）
    prompts = {}
    if prompt_path and os.path.exists(prompt_path):
        import json
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        print(f"✅ Loaded CuPL prompts from {prompt_path}")

    with torch.no_grad():
        clip_weights = []
        min_len = 1000

        for classname in tqdm(classnames, desc="Processing classes"):
            classname_clean = classname.replace('_', ' ')

            # 基础模板
            template_texts = [t.format(classname_clean) for t in template]

            # 添加CuPL prompts（如果有）
            texts = template_texts
            if classname in prompts:
                texts += prompts[classname]

            # 编码
            texts_token = clip.tokenize(texts, truncate=True).cuda()
            class_embeddings = clip_model.encode_text(texts_token)

            if norm:
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            min_len = min(min_len, len(class_embeddings))
            clip_weights.append(class_embeddings)

        # 统一长度（取最短）
        for i in range(len(clip_weights)):
            clip_weights[i] = clip_weights[i][:min_len]

        clip_weights = torch.stack(clip_weights, dim=0).cuda()
        print(f"\n✅ Text prompts shape: {clip_weights.shape}")
        print(f"   Classes: {clip_weights.shape[0]}")
        print(f"   Prompts per class: {clip_weights.shape[1]}")
        print(f"   Feature dim: {clip_weights.shape[2]}")

    # 保存
    save_path = os.path.join(cache_dir, "text_weights_cupl_t_all.pt")
    torch.save(clip_weights, save_path)
    print(f"✅ Saved to: {save_path}")
    print("=" * 60 + "\n")

    return clip_weights


def extract_few_shot_feature_all(cache_dir, clip_model, train_loader,
                                 shots, augment_epoch=1, norm=True):
    """
    提取训练集的所有图像特征

    Args:
        cache_dir: 保存路径
        clip_model: CLIP模型
        train_loader: 训练数据加载器
        shots: 样本数
        augment_epoch: 数据增强轮数
        norm: 是否归一化
    """
    print("\n" + "=" * 60)
    print(f"Extracting {shots}-shot training features...")
    print("=" * 60)

    with torch.no_grad():
        vecs = []
        labels = []

        for epoch in range(augment_epoch):
            print(f"Augment epoch {epoch + 1}/{augment_epoch}")
            for image, target in tqdm(train_loader, desc="Processing batches"):
                image, target = image.cuda(), target.cuda()
                image_features = clip_model.encode_image(image)

                if norm:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                vecs.append(image_features)
                labels.append(target)

        vecs = torch.cat(vecs)
        labels = torch.cat(labels)

        print(f"✅ Extracted features shape: {vecs.shape}")
        print(f"✅ Labels shape: {labels.shape}")

    # 保存
    vecs_path = os.path.join(cache_dir, f"{shots}_vecs_f.pt")
    labels_path = os.path.join(cache_dir, f"{shots}_labels_f.pt")

    torch.save(vecs, vecs_path)
    torch.save(labels, labels_path)

    print(f"✅ Saved features to: {vecs_path}")
    print(f"✅ Saved labels to: {labels_path}")
    print("=" * 60 + "\n")

    return vecs, labels


def extract_test_feature(cache_dir, split, clip_model, test_loader, norm=True):
    """
    提取测试集特征

    Args:
        cache_dir: 保存路径
        split: 'val' 或 'test'
        clip_model: CLIP模型
        test_loader: 测试数据加载器
        norm: 是否归一化
    """
    print(f"\nExtracting {split} features...")

    features, labels = [], []

    with torch.no_grad():
        for images, target in tqdm(test_loader, desc=f"Processing {split}"):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)

            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features)
            labels.append(target)

    features = torch.cat(features)
    labels = torch.cat(labels)

    # 保存
    feat_path = os.path.join(cache_dir, f"{split}_f.pt")
    label_path = os.path.join(cache_dir, f"{split}_l.pt")

    torch.save(features, feat_path)
    torch.save(labels, label_path)

    print(f"✅ Saved {split} features: {feat_path}")
    print(f"   Shape: {features.shape}\n")

    return features, labels


if __name__ == '__main__':
    # ===== 配置 =====
    dataset_name = 'dtd'
    shots = 16
    root_path = './data'  # 你的数据集路径
    cache_base = './caches'

    # ===== 加载CLIP =====
    print("\n" + "=" * 60)
    print("Loading CLIP model...")
    print("=" * 60)

    clip_model, preprocess = clip.load("RN50")
    clip_model.eval().cuda()

    print("✅ CLIP model loaded!")
    print("=" * 60 + "\n")

    # ===== 加载数据集 =====
    print(f"Loading {dataset_name} dataset ({shots}-shot)...")

    dataset = DescribableTextures(root_path, shots)
    dataset._classnames = sorted(dataset.classnames)

    cache_dir = os.path.join(cache_base, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"✅ Dataset loaded!")
    print(f"   Classes: {len(dataset.classnames)}")
    print(f"   Train samples: {len(dataset.train_x)}")
    print(f"   Test samples: {len(dataset.test)}")
    print(f"   Cache dir: {cache_dir}\n")

    # ===== 数据加载器 =====
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader = build_data_loader(
        data_source=dataset.train_x,
        batch_size=256,
        tfm=train_transform,
        is_train=True,
        shuffle=False
    )

    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=64,
        tfm=preprocess,
        is_train=False,
        shuffle=False
    )

    # ===== 1. 提取训练集特征 =====
    train_vecs, train_labels = extract_few_shot_feature_all(
        cache_dir=cache_dir,
        clip_model=clip_model,
        train_loader=train_loader,
        shots=shots,
        augment_epoch=1,
        norm=True
    )

    # ===== 2. 提取测试集特征 =====
    test_features, test_labels = extract_test_feature(
        cache_dir=cache_dir,
        split='test',
        clip_model=clip_model,
        test_loader=test_loader,
        norm=True
    )

    # ===== 3. 提取文本prompt特征 =====
    text_prompts_all = extract_text_feature_all(
        cache_dir=cache_dir,
        classnames=dataset.classnames,
        clip_model=clip_model,
        template=dataset.template,
        prompt_path=None,  # 如果有CuPL prompt文件，填路径
        norm=True
    )

    print("\n" + "=" * 60)
    print("✅ Feature extraction completed!")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")
    print(f"\nExtracted files:")
    print(f"  - {shots}_vecs_f.pt       (training features)")
    print(f"  - {shots}_labels_f.pt     (training labels)")
    print(f"  - test_f.pt               (test features)")
    print(f"  - test_l.pt               (test labels)")
    print(f"  - text_weights_cupl_t_all.pt  (prompt features)")
    print("=" * 60)
    print("\n🎉 Now you can run: python Prompt.py --dataset dtd --shots 16")