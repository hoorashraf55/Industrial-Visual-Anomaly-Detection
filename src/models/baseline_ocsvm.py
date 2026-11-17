import numpy as np
import torch
from skimage.feature import hog
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class BaselineOCSVM:
    """
    Classical ML Baseline: Patch-based HOG Feature Extraction + One-Class SVM.
    Produces both image-level anomaly scores and spatial decision scoring maps.
    """
    def __init__(self, patch_size: int = 32, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        self.patch_size = patch_size
        self.scaler = StandardScaler()
        self.ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def _extract_patches_hog(self, images: torch.Tensor) -> tuple:
        # images: (B, 3, H, W) -> convert to numpy grayscale
        imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()
        imgs_gray = imgs_np.mean(axis=-1)
        B, H, W = imgs_gray.shape
        gh, gw = H // self.patch_size, W // self.patch_size

        all_feats = []
        for b in range(B):
            img_feats = []
            for i in range(gh):
                for j in range(gw):
                    patch = imgs_gray[b, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size]
                    feat = hog(patch, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True)
                    img_feats.append(feat)
            all_feats.append(img_feats)
        return np.array(all_feats), (gh, gw)  # (B, gh*gw, D)

    def fit(self, train_loader):
        patch_features = []
        for images, _, _, _ in train_loader:
            feats, _ = self._extract_patches_hog(images)
            patch_features.append(feats.reshape(-1, feats.shape[-1]))
        X_train = np.vstack(patch_features)
        X_scaled = self.scaler.fit_transform(X_train)
        self.ocsvm.fit(X_scaled)

    def predict(self, loader) -> tuple:
        all_scores, all_maps, all_labels, all_masks = [], [], [], []
        for images, masks, labels, _ in loader:
            B, _, H, W = images.shape
            feats, (gh, gw) = self._extract_patches_hog(images)
            flat_feats = feats.reshape(-1, feats.shape[-1])
            scaled_feats = self.scaler.transform(flat_feats)
            
            # Negative decision_function = higher anomaly score
            raw_scores = -self.ocsvm.decision_function(scaled_feats)
            patch_maps = raw_scores.reshape(B, 1, gh, gw)
            
            # Upsample patch map to (H, W)
            upsampled = torch.nn.functional.interpolate(
                torch.tensor(patch_maps, dtype=torch.float32), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1).numpy()

            img_scores = upsampled.reshape(B, -1).max(axis=1)
            all_scores.extend(img_scores)
            all_maps.append(upsampled)
            all_labels.extend(labels.numpy())
            all_masks.append(masks.squeeze(1).numpy())

        return np.array(all_labels), np.array(all_scores), np.vstack(all_masks), np.vstack(all_maps)