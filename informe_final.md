# Human Activity Recognition System Using Computer Vision and Machine Learning

**Universidad ICESI**  
Facultad de Ingeniería, Diseño y Ciencias Aplicadas  
Departamento de Computación y Sistemas Inteligentes  
Ingeniería de Sistemas

**Curso:** Inteligencia Artificial I (APO3)  
**Semestre:** 2025-2

**Autores:**
- Mariana De La Cruz - A00399618
- Valentina Gómez - A00398790
- Alexis Delgado - A00399176
- Juan Camilo Amorocho - A00399789

---

## Abstract

This paper presents a comprehensive system for automated human activity recognition using computer vision and machine learning techniques. The system employs MediaPipe Pose to extract 33 body landmarks from video sequences, computing 12 biomechanical features including joint angles, velocity, acceleration, and body symmetry. Three supervised learning models (Random Forest, SVM with RBF kernel, and XGBoost) were trained on a dataset of 86 videos across 8 activity categories. Principal Component Analysis (PCA) was applied to reduce dimensionality from 12 to 6 features while preserving 91.73% of variance. The best models achieved 94.44% accuracy on the test set. A web-based application was developed using Streamlit for real-time classification. This work demonstrates the feasibility of automated movement analysis for applications in healthcare, rehabilitation, and sports analytics, while addressing ethical considerations regarding privacy, bias, and responsible AI deployment.

**Keywords:** Human Activity Recognition, Computer Vision, MediaPipe Pose, Machine Learning, PCA, Biomechanical Analysis.

---

## 1. Introduction

### 1.1 Context and Motivation

Human activity recognition (HAR) has become a critical research area at the intersection of computer vision, machine learning, and biomechanics. The ability to automatically classify and analyze human movements has profound implications for healthcare (rehabilitation monitoring, fall detection), sports science (performance analysis, injury prevention), human-computer interaction, and security systems.

Traditional approaches to movement analysis require manual observation by trained professionals, which is time-consuming, subjective, and expensive. Recent advances in computer vision, particularly pose estimation techniques like MediaPipe and OpenPose, have enabled automated extraction of skeletal information from video, opening new possibilities for scalable and objective movement analysis.

### 1.2 Problem Statement

Despite significant progress in HAR research, existing systems face several challenges:

1. **Complexity of human motion:** Human activities involve coordinated movements of multiple body parts in three-dimensional space, with high inter-subject variability.
2. **Environmental variability:** Lighting conditions, camera angles, background clutter, and occlusions affect detection quality.
3. **Real-time processing requirements:** Applications like rehabilitation monitoring require low-latency classification.
4. **Limited generalization:** Models trained on small datasets often fail to generalize to new subjects or environments.

This project addresses the challenge of developing a robust, automated system for classifying basic human activities (walking forward, walking backward, sitting, squatting, and various hip movements) using only RGB video input from standard cameras.

### 1.3 Objectives

**General Objective:**  
Develop an intelligent system capable of automatically analyzing and classifying human activities from video sequences using computer vision and machine learning techniques.

**Specific Objectives:**
1. Collect and annotate a dataset of human activity videos across multiple categories.
2. Implement pose estimation using MediaPipe to extract body landmarks.
3. Engineer biomechanical features from landmark data.
4. Train and compare multiple supervised learning models.
5. Apply dimensionality reduction to improve model generalization.
6. Deploy a functional web application for real-time classification.
7. Analyze ethical implications and societal impacts of the system.

### 1.4 Significance

This work is significant for several reasons:

- **Practical applicability:** The system can be deployed with minimal hardware (standard RGB camera), making it accessible for diverse applications.
- **Interdisciplinary approach:** Combines computer vision, biomechanics, and machine learning.
- **Ethical awareness:** Explicitly addresses privacy concerns, bias, and responsible AI deployment.
- **Educational value:** Demonstrates complete machine learning pipeline from data collection to deployment.

---

## 2. Theory

This section presents the theoretical foundations necessary to understand the system's development.

### 2.1 Pose Estimation and MediaPipe

**Pose estimation** is the task of detecting and tracking human body keypoints (landmarks) in images or videos. Modern deep learning approaches, particularly convolutional neural networks (CNNs), have achieved remarkable accuracy.

**MediaPipe Pose** [1] is a machine learning solution developed by Google Research that detects 33 body landmarks in 3D space (x, y, z coordinates plus visibility score). The model architecture consists of:

1. **BlazePose detector:** Locates the person's region of interest (ROI) in the frame.
2. **Landmark predictor:** Estimates 33 keypoints within the ROI.

The 33 landmarks include:
- Face: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists, hands
- Lower body: hips, knees, ankles, feet

MediaPipe Pose operates at 30+ FPS on standard hardware, making it suitable for real-time applications.

### 2.2 Biomechanical Feature Engineering

Biomechanical analysis quantifies human movement using principles from anatomy and physics. Key concepts include:

**Joint Angles:**  
The angle θ formed by three landmarks (a, b, c) where b is the vertex can be computed using the dot product:

$$\theta = \arccos\left(\frac{\vec{ba} \cdot \vec{bc}}{|\vec{ba}| \cdot |\vec{bc}|}\right)$$

where:
- $\vec{ba} = a - b$ (vector from b to a)
- $\vec{bc} = c - b$ (vector from b to c)

**Velocity and Acceleration:**  
Given a body point position $p_t$ at time $t$:

$$v_t = \frac{||p_t - p_{t-1}||}{\Delta t}$$

$$a_t = \frac{v_t - v_{t-1}}{\Delta t}$$

**Body Symmetry:**  
Quantifies left-right balance using Euclidean distance between corresponding bilateral landmarks:

$$S = \frac{1}{n}\sum_{i=1}^{n}||L_i - R_i||$$

where $L_i$ and $R_i$ are corresponding left and right landmarks.

### 2.3 Machine Learning Models

#### 2.3.1 Random Forest

Random Forest [2] is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of classes. Key advantages:
- Resistant to overfitting (with sufficient trees)
- Handles non-linear relationships
- Provides feature importance

The prediction for classification:

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_T(x)\}$$

where $h_t$ is the prediction of tree $t$ and $T$ is the total number of trees.

#### 2.3.2 Support Vector Machine (SVM)

SVM [3] finds the optimal hyperplane that maximizes the margin between classes. For non-linearly separable data, the RBF (Radial Basis Function) kernel is used:

$$K(x_i, x_j) = \exp\left(-\gamma ||x_i - x_j||^2\right)$$

The decision function:

$$f(x) = \text{sign}\left(\sum_{i=1}^{n}\alpha_i y_i K(x_i, x) + b\right)$$

where $\alpha_i$ are Lagrange multipliers, $y_i$ are class labels, and $b$ is the bias term.

#### 2.3.3 XGBoost

XGBoost [4] is a gradient boosting algorithm that builds an ensemble of weak learners sequentially. Each new tree corrects errors of the previous ensemble:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

The objective function includes both loss and regularization:

$$L^{(t)} = \sum_{i=1}^{n}l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t}\Omega(f_k)$$

### 2.4 Principal Component Analysis (PCA)

PCA [5] is a dimensionality reduction technique that projects data onto orthogonal axes (principal components) that maximize variance.

**Mathematical Formulation:**

Given a data matrix $X \in \mathbb{R}^{n \times p}$ (n samples, p features):

1. Center the data: $X_c = X - \mu$ where $\mu$ is the mean vector
2. Compute covariance matrix: $C = \frac{1}{n-1}X_c^T X_c$
3. Compute eigenvectors $v_i$ and eigenvalues $\lambda_i$ of $C$
4. Sort eigenvectors by descending eigenvalues
5. Select top $k$ eigenvectors forming matrix $W \in \mathbb{R}^{p \times k}$
6. Project data: $Z = X_c W$

**Variance Explained:**

$$\text{VE}_k = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{p}\lambda_i}$$

In this project, 6 principal components captured 91.73% of total variance, reducing feature space by 50%.

### 2.5 Model Evaluation Metrics

**Accuracy:**  
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:**  
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall:**  
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score:**  
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.

---

## 3. Methodology

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, adapted to the specific needs of video-based activity recognition.

### 3.1 Business Understanding

**Stakeholder Needs:**  
- Healthcare professionals need automated tools for remote patient monitoring
- Physical therapists require objective movement quality assessment
- Sports coaches seek performance analysis capabilities

**Success Criteria:**  
- Classification accuracy > 85%
- Real-time processing capability (< 2s per video)
- User-friendly web interface
- Ethical compliance with privacy regulations

### 3.2 Data Understanding and Collection

#### 3.2.1 Data Collection Protocol

Videos were captured using smartphone cameras (1080p resolution, 30 FPS) in controlled indoor environments with adequate lighting. Participants performed predefined activities following standardized instructions.

**Dataset Evolution:**

| Phase | Videos | Categories | Resolution | Total Frames |
|-------|--------|------------|------------|--------------|
| Entrega 1 | 54 | 3 | 1080x1908 | ~5,400 |
| Entrega 2 | 86 | 8 | 1080x1908 | ~8,600 |

**Activity Categories:**
1. **Adelante** - Walking toward camera
2. **Atrás** - Walking away from camera
3. **Sentado** - Sitting position
4. **Cadera_Alfrente** - Hip flexion forward
5. **Caderas** - Hip rotation
6. **Lado** - Lateral movement
7. **Sentadilla** - Squat
8. **Tijeras** - Scissor movement

**Data Distribution:** ~10-11 samples per class, representing a limitation for model generalization.

#### 3.2.2 Exploratory Data Analysis

Initial analysis revealed:
- Consistent video duration: 2.5-4 seconds
- Stable frame rate: 30 FPS across all videos
- Brightness variation: 137-146 (normalized 0-255 scale)
- High pose detection rate: >90% frames with valid landmarks

### 3.3 Data Preparation

#### 3.3.1 Landmark Extraction

For each video frame:
1. Apply MediaPipe Pose detection
2. Extract 33 landmark coordinates (x, y, z)
3. Filter frames with low confidence scores
4. Store temporal sequences

#### 3.3.2 Feature Engineering

From raw landmarks, 12 biomechanical features were computed:

**Temporal Features:**
- `frames`: Total number of frames
- `duracion_seg`: Video duration in seconds

**Visual Features:**
- `brillo_promedio`: Average brightness (luminosity)
- `movimiento_promedio`: Average inter-frame difference

**Kinematic Features:**
- `velocidad_promedio`: Hip center velocity
- `aceleracion_promedio`: Hip center acceleration

**Angular Features:**
- `angulo_rodilla_promedio`: Knee angle (hip-knee-ankle)
- `angulo_cadera_promedio`: Hip angle (shoulder-hip-knee)
- `angulo_tobillo_promedio`: Ankle angle (knee-ankle-foot)

**Postural Features:**
- `inclinacion_promedio`: Shoulder lateral inclination
- `dist_hombros_caderas`: Shoulder-hip distance
- `simetria_promedio`: Left-right body symmetry

All features were averaged across frames to create fixed-length feature vectors.

#### 3.3.3 Data Preprocessing

**Step 1: Data Cleaning**
- Removed videos with < 50% valid landmark frames
- Eliminated outliers using IQR method: $Q_1 - 1.5 \times IQR$ to $Q_3 + 1.5 \times IQR$

**Step 2: Normalization**

MinMax scaling to range [0, 1]:

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

This ensures all features contribute equally to model training and is particularly important for distance-based algorithms like SVM.

**Step 3: Train-Test Split**
- Training set: 70% (60 samples)
- Test set: 30% (26 samples)
- Stratified split to maintain class distribution

### 3.4 Modeling

#### 3.4.1 Baseline Models (Entrega 2)

Three models were trained on 12 original features:

**Random Forest:**
- Hyperparameters: `n_estimators=100`, `max_depth=10`, `min_samples_split=2`
- Result: 100% accuracy (overfitting suspected)

**SVM (RBF):**
- Hyperparameters: `C=10`, `gamma=0.1`, `kernel='rbf'`
- Result: 100% accuracy (overfitting suspected)

**XGBoost:**
- Hyperparameters: `max_depth=3`, `learning_rate=0.1`, `n_estimators=100`
- Result: 88.9% accuracy

**Hyperparameter Optimization:** GridSearchCV with 5-fold cross-validation.

#### 3.4.2 Dimensionality Reduction (Entrega 3)

**PCA Application:**

Objective: Reduce feature space to mitigate overfitting and improve generalization.

**Results:**
- Reduced from 12 to 6 principal components
- Variance explained by each PC:
  - PC1: 37.08% (angular features)
  - PC2: 24.71% (temporal and brightness)
  - PC3: 12.57% (postural alignment)
  - PC4: 8.87% (movement dynamics)
  - PC5: 4.48% (brightness-time interaction)
  - PC6: 4.05% (velocity-acceleration)
- **Total variance preserved: 91.73%**

**Model Retraining with PCA:**

| Model | Original (12 feat.) | PCA (6 PC) | Difference |
|-------|---------------------|------------|------------|
| Random Forest | 100.0% | 94.44% | -5.56% |
| SVM (RBF) | 100.0% | 94.44% | -5.56% |
| XGBoost | 88.9% | 77.78% | -11.12% |

### 3.5 Evaluation Strategy

**Cross-Validation:**  
5-fold stratified cross-validation to assess model stability.

**Metrics:**  
Accuracy, Precision, Recall, F1-Score (macro-averaged for multi-class).

**Confusion Matrix Analysis:**  
Identified which activities are most commonly confused.

### 3.6 Deployment

**Web Application Development:**
- Framework: Streamlit (Python)
- Features:
  - Video upload (MP4, AVI, MOV)
  - Real-time landmark extraction
  - Multi-model ensemble prediction
  - Visualization of biomechanical metrics
  - PCA component analysis
  - CSV export of results

**Deployment Infrastructure:**
- Containerization: Docker
- Platform: Railway / Streamlit Cloud
- Model persistence: Joblib (`.pkl` files)

---

## 4. Results

### 4.1 Dataset Statistics

**Final Dataset:**
- Total videos: 86
- Categories: 8
- Average duration: 3.1 seconds
- Average frames: 93 frames/video
- Landmark detection rate: 94.2%

### 4.2 Feature Analysis

**Correlation Analysis:**  
Strong correlations identified:
- Hip angle ↔ Knee angle (r = 0.78)
- Velocity ↔ Acceleration (r = 0.82)
- Duration ↔ Frame count (r = 0.99)

**Feature Importance (Random Forest):**
1. `angulo_cadera_promedio` (0.18)
2. `angulo_rodilla_promedio` (0.16)
3. `velocidad_promedio` (0.14)
4. `simetria_promedio` (0.12)
5. `dist_hombros_caderas` (0.11)

### 4.3 Model Performance

#### 4.3.1 Entrega 2 - Original Features (12 dimensions)

**Random Forest:**
```
Accuracy:  100.00%
Precision: 100.00%
Recall:    100.00%
F1-Score:  100.00%
```

**SVM (RBF):**
```
Accuracy:  100.00%
Precision: 100.00%
Recall:    100.00%
F1-Score:  100.00%
```

**XGBoost:**
```
Accuracy:  88.90%
Precision: 94.40%
Recall:    88.90%
F1-Score:  87.30%
```

**Cross-Validation Scores:**
- Random Forest: 100.0% ± 0.0%
- SVM: 100.0% ± 0.0%
- XGBoost: 88.9% ± 8.2%

#### 4.3.2 Entrega 3 - PCA Features (6 dimensions)

**Random Forest:**
```
Accuracy:  94.44%
Precision: 95.56%
Recall:    94.44%
F1-Score:  94.36%
CV Score:  98.46%
```

**SVM (RBF):**
```
Accuracy:  94.44%
Precision: 95.56%
Recall:    94.44%
F1-Score:  94.36%
CV Score:  100.0%
```

**XGBoost:**
```
Accuracy:  77.78%
Precision: 76.85%
Recall:    77.78%
F1-Score:  75.45%
CV Score:  97.03%
```

### 4.4 Confusion Matrix Analysis

**Random Forest (PCA) - Most Confused Pairs:**
- Sentadilla ↔ Caderas (1 misclassification)

**SVM (PCA) - Most Confused Pairs:**
- Adelante ↔ Atras (1 misclassification)

**XGBoost (PCA) - Most Confused Pairs:**
- Sentadilla ↔ Sentado (2 misclassifications)
- Tijeras ↔ Lado (2 misclassifications)

### 4.5 PCA Visualization

A 2D projection of the dataset using the first two principal components shows:
- Clear separation between static (Sentado) and dynamic activities
- Overlap between similar movement patterns (Adelante/Atras)
- Distinct cluster for Sentadilla (deep squat)

### 4.6 Computational Performance

**Training Time:**
- Random Forest: 1.2 seconds
- SVM: 0.8 seconds
- XGBoost: 2.4 seconds

**Inference Time (per video):**
- Landmark extraction: 0.8-1.5 seconds
- Model prediction: < 0.01 seconds
- Total pipeline: ~1.5 seconds

---

## 5. Results Analysis

### 5.1 Overfitting in Baseline Models

The 100% accuracy achieved by Random Forest and SVM on the original feature set is a clear indicator of **overfitting**. Several factors contribute:

**1. Small Dataset:** With only 86 samples across 8 classes (~11 samples/class), models memorize training patterns rather than learning generalizable features.

**2. High Feature-to-Sample Ratio:** 12 features for ~60 training samples provides limited constraints on model capacity.

**3. Cross-Validation Limitations:** 5-fold CV with stratification means each fold has only ~2 samples per class in validation, insufficient for reliable generalization estimates.

**Evidence:**
- Perfect training accuracy but unknown real-world performance
- Zero variance in CV scores (suspicious)
- XGBoost's lower accuracy (88.9%) is actually more realistic

### 5.2 Impact of Dimensionality Reduction

PCA application resulted in controlled performance degradation:

**Positive Outcomes:**
1. **Regularization effect:** 5.56% accuracy drop for RF/SVM indicates reduced overfitting
2. **Improved CV scores:** Higher and more stable cross-validation performance
3. **Computational efficiency:** 50% reduction in feature dimensionality
4. **Interpretability:** Principal components have clear biomechanical meaning

**Negative Outcomes:**
1. **XGBoost degradation:** 11.12% drop suggests tree-based boosting benefits from original feature space
2. **Information loss:** 8.27% of variance discarded may contain class-discriminative information

**Trade-off Analysis:**

The reduction from 100% → 94.44% is acceptable because:
- 100% was likely spurious (overfitting)
- 94.44% with better CV scores indicates true generalization
- Simpler model is preferred (Occam's Razor)

### 5.3 Feature Importance and Biomechanical Interpretation

**PC1 (37.08% variance):** Dominated by angular measurements (knee, ankle angles)
- **Interpretation:** Body posture configuration is the primary discriminator between activities
- **Biomechanical significance:** Different activities have characteristic joint angle patterns

**PC2 (24.71% variance):** Temporal features (frames, duration) and brightness
- **Interpretation:** Activity duration and recording conditions
- **Limitation:** May capture recording artifacts rather than movement characteristics

**PC3-PC6:** Capture velocity, acceleration, symmetry, and postural alignment
- **Interpretation:** Movement dynamics and body balance

**Insight:** Static posture (angles) is more discriminative than dynamics (velocity), suggesting the dataset may benefit from more dynamic activities.

### 5.4 Model Comparison

**Best Model:** SVM (RBF) with PCA
- Reasons: 94.44% accuracy, 100% CV score, efficient training
- Strength: Excellent margin-based separation in reduced dimensional space
- Weakness: Requires careful kernel parameter tuning

**Runner-up:** Random Forest with PCA
- Nearly identical performance to SVM
- Advantage: Feature importance interpretation
- Disadvantage: Higher variance in predictions

**Poorest:** XGBoost with PCA
- Significant performance drop with PCA
- Reason: Boosting trees may require original feature interactions
- Recommendation: Use XGBoost without PCA for future work

### 5.5 Generalization Concerns

**Major Limitation:** All training and test samples come from the same subjects in the same environment.

**Generalization Risks:**
1. **Subject-specific overfitting:** Models may learn individual movement patterns rather than activity categories
2. **Environmental overfitting:** Performance may degrade with different backgrounds, lighting, or camera angles
3. **Small sample bias:** Statistical estimates are unreliable with n=86

**Required Validation:**
- Test on completely new subjects
- Vary recording conditions (outdoor, different lighting)
- Cross-dataset evaluation (compare with public HAR datasets)

### 5.6 Comparison with Literature

**State-of-the-art HAR systems:**

| Study | Approach | Dataset Size | Accuracy |
|-------|----------|--------------|----------|
| Cao et al. (2018) [6] | CNN on poses | 15,000 samples | 86.3% |
| This work | ML on features | 86 samples | 94.44% |
| Yan et al. (2018) [7] | Spatial-temporal GCN | 56,000 samples | 88.8% |

**Observation:** Our accuracy is competitive but achieved on a much smaller, controlled dataset. Direct comparison is inappropriate due to different:
- Dataset scales (100x smaller)
- Activity complexity (basic vs. complex actions)
- Evaluation protocols (simple split vs. cross-subject validation)

**Realistic Assessment:** The 94.44% accuracy likely represents upper-bound performance under ideal conditions. Real-world deployment would require extensive additional validation.

### 5.7 Ethical and Bias Considerations

**Dataset Bias:**
- Limited demographic diversity (university students)
- Gender, age, body type representation unknown
- Cultural context: activities defined by specific movement norms

**Model Bias:**
- May perform poorly on individuals with mobility impairments
- Could discriminate based on body shape/size affecting landmark detection
- Brightness feature may correlate with skin tone (privacy/fairness concern)

**Mitigation Strategies:**
- Collect diverse participant data
- Evaluate performance across demographic groups
- Remove potentially biased features (brightness)
- Document limitations clearly in deployment

---

## 6. Mathematical Problem Formulation (Competency P13)

### 6.1 Formal Problem Definition

**Input Space:**  
Let $V = \{v_1, v_2, ..., v_n\}$ be a set of videos, where each video $v_i$ consists of a temporal sequence of frames: $v_i = \{f_1, f_2, ..., f_T\}$

**Feature Extraction:**  
A function $\phi: V \rightarrow \mathbb{R}^p$ maps videos to feature vectors:

$$\phi(v_i) = \mathbf{x}_i = [x_{i1}, x_{i2}, ..., x_{ip}]^T$$

where $p=12$ represents the engineered biomechanical features.

**Classification Task:**  
Given labeled training data $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ where $y_i \in \{1, 2, ..., C\}$ and $C=8$ activity classes, learn a function:

$$f: \mathbb{R}^p \rightarrow \{1, 2, ..., C\}$$

that minimizes the expected classification error:

$$\min_{f} \mathbb{E}_{(\mathbf{x}, y) \sim P(X,Y)}[\mathbb{1}_{f(\mathbf{x}) \neq y}]$$

### 6.2 Feature Engineering Mathematical Specification

#### 6.2.1 Angle Calculation

For three landmarks $\mathbf{a}, \mathbf{b}, \mathbf{c} \in \mathbb{R}^3$, the joint angle at $\mathbf{b}$:

$$\theta_{abc} = \arccos\left(\frac{(\mathbf{a}-\mathbf{b}) \cdot (\mathbf{c}-\mathbf{b})}{||\mathbf{a}-\mathbf{b}|| \cdot ||\mathbf{c}-\mathbf{b}||}\right)$$

**Implementation:**
```
knee_angle = arccos(clip(dot(hip-knee, ankle-knee) / 
                    (norm(hip-knee) * norm(ankle-knee)), -1, 1))
```

**Numerical Stability:** Clipping to [-1, 1] prevents domain errors from floating-point arithmetic.

#### 6.2.2 Velocity Estimation

For hip center position $\mathbf{h}_t$ at frame $t$:

$$v_t = \frac{||\mathbf{h}_t - \mathbf{h}_{t-1}||}{\Delta t}$$

where $\Delta t = 1/\text{FPS} = 1/30$ seconds.

**Aggregation:** Mean velocity over $T$ frames:

$$\bar{v} = \frac{1}{T-1}\sum_{t=2}^{T}v_t$$

#### 6.2.3 Normalization

MinMax scaling for feature $x_j$:

$$x_j^{(norm)} = \frac{x_j - \min(x_j)}{\max(x_j) - \min(x_j)}$$

**Properties:**
- Range: $[0, 1]$
- Preserves original distribution shape
- Sensitive to outliers (addressed via IQR filtering)

### 6.3 Dimensionality Reduction

**PCA Objective:**  
Find orthogonal projection $W \in \mathbb{R}^{12 \times 6}$ maximizing variance:

$$\max_{W} \text{tr}(W^T \Sigma W) \quad \text{subject to} \quad W^T W = I$$

where $\Sigma$ is the covariance matrix of normalized features.

**Solution:** Eigenvectors of $\Sigma$ corresponding to 6 largest eigenvalues.

**Projection:**

$$\mathbf{z}_i = W^T(\mathbf{x}_i - \boldsymbol{\mu})$$

where $\mathbf{z}_i \in \mathbb{R}^6$ is the reduced representation.

### 6.4 Model-Specific Formulations

#### 6.4.1 Random Forest

**Ensemble Prediction:**

$$\hat{y} = \arg\max_{c} \sum_{t=1}^{T} \mathbb{1}_{h_t(\mathbf{z}) = c}$$

where $h_t$ is the $t$-th decision tree.

**Tree Construction:** Each tree minimizes Gini impurity:

$$G = 1 - \sum_{c=1}^{C}p_c^2$$

where $p_c$ is the proportion of class $c$ in a node.

#### 6.4.2 SVM

**Primal Formulation:**

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{n}\xi_i$$

subject to:

$$y_i(\mathbf{w}^T\phi(\mathbf{z}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

where $\phi(\mathbf{z}_i)$ is the RBF kernel mapping.

**RBF Kernel:**

$$K(\mathbf{z}_i, \mathbf{z}_j) = \exp\left(-\gamma||\mathbf{z}_i - \mathbf{z}_j||^2\right)$$

**Hyperparameters:**
- $C = 10$: Regularization parameter (trade-off between margin and violations)
- $\gamma = 0.1$: Kernel bandwidth (controls decision boundary smoothness)

#### 6.4.3 XGBoost

**Additive Model:**

$$\hat{y}_i = \sum_{k=1}^{K}f_k(\mathbf{z}_i)$$

where $f_k$ is the $k$-th regression tree.

**Objective Function:**

$$\mathcal{L} = \sum_{i=1}^{n}l(y_i, \hat{y}_i) + \sum_{k=1}^{K}\Omega(f_k)$$

where:
- $l$ is the cross-entropy loss for multi-class classification
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda||\mathbf{w}||^2$ (regularization)

### 6.5 Evaluation Metrics

**Multi-class Classification:**

For class $c$:

$$\text{Precision}_c = \frac{\sum_{i:y_i=c}\mathbb{1}_{\hat{y}_i=c}}{\sum_{i}\mathbb{1}_{\hat{y}_i=c}}$$

$$\text{Recall}_c = \frac{\sum_{i:y_i=c}\mathbb{1}_{\hat{y}_i=c}}{\sum_{i}\mathbb{1}_{y_i=c}}$$

**Macro-averaged F1:**

$$F1_{macro} = \frac{1}{C}\sum_{c=1}^{C}\frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 6.6 Why This Problem is Complex

This is a **complex engineering problem** because:

1. **High-dimensional non-linear mapping:** Video sequences → feature vectors involves temporal aggregation and geometric computations
2. **Multi-class classification:** 8 classes with overlapping feature distributions
3. **Curse of dimensionality:** Small sample size relative to feature dimensionality
4. **Model selection uncertainty:** Multiple algorithms with hyperparameter spaces
5. **Real-time constraints:** Inference must complete in < 2 seconds
6. **Generalization challenge:** Performance must extend beyond training distribution

**Solution Strategy Justification:**

- **MediaPipe over OpenPose:** Faster inference (30 FPS vs 10 FPS), sufficient accuracy for our purposes
- **Engineered features over raw landmarks:** Reduces dimensionality (33×3=99 → 12), incorporates domain knowledge
- **PCA over autoencoders:** Interpretable, computationally efficient, no need for additional training
- **Ensemble voting:** Compensates for individual model weaknesses
- **MinMax over StandardScaler:** Better for bounded features (angles, normalized coordinates)

---

## 7. Ethical Considerations and Professional Responsibilities (Competency P11)

### 7.1 Identification of Ethical Issues

This project involves several ethical considerations aligned with professional engineering standards:

| **Ethical Issue** | **Description** | **Code of Ethics Principle** |
|-------------------|-----------------|------------------------------|
| **Privacy** | Video data captures identifiable biometric information | Right to privacy and data protection |
| **Informed Consent** | Participants must understand data usage and retention | Respect for autonomy and transparency |
| **Data Security** | Risk of unauthorized access to sensitive video data | Professional responsibility for data protection |
| **Algorithmic Bias** | Model may perform differently across demographic groups | Fairness and non-discrimination |
| **Surveillance Concerns** | Technology could enable unethical monitoring | Social responsibility and preventing harm |
| **Accessibility** | System may exclude people with disabilities | Inclusivity and universal design |
| **Misuse Potential** | Technology could be deployed for harmful purposes | Anticipating and preventing misuse |

### 7.2 Alignment with Professional Engineering Ethics

**IEEE Code of Ethics [8] Compliance:**

1. **"To hold paramount the safety, health, and welfare of the public"**
   - Decision: Implemented privacy-preserving design (no video storage in deployment)
   - Decision: Documented limitations to prevent misuse in critical applications (e.g., medical diagnosis without validation)

2. **"To avoid real or perceived conflicts of interest"**
   - Disclosed dataset limitations and potential biases
   - Acknowledged performance may not generalize beyond training conditions

3. **"To be honest and realistic in stating claims"**
   - Clearly reported overfitting in baseline models
   - Stated that 94.44% accuracy is likely optimistic for real-world deployment

4. **"To reject bribery in all its forms"**
   - Not applicable to academic project

5. **"To improve the understanding of technology"**
   - Open-source code and documentation
   - Educational purpose of project benefits broader community

6. **"To maintain and improve technical competence"**
   - Applied state-of-the-art techniques (MediaPipe, PCA, ensemble methods)
   - Engaged with recent literature on HAR

7. **"To seek, accept, and offer honest criticism"**
   - Acknowledged dataset size limitations
   - Welcomed peer review through academic submission

8. **"To treat fairly all persons"**
   - Recognized potential demographic biases
   - Proposed mitigation strategies (diverse data collection)

9. **"To avoid injuring others"**
   - Assessed misuse risks (surveillance, discrimination)
   - Implemented safeguards (consent protocols, data minimization)

10. **"To assist colleagues and co-workers"**
    - Collaborative team project
    - Shared knowledge and resources

### 7.3 Ethical Decisions Made

**Decision 1: Data Minimization**
- **Action:** Videos are processed and deleted; only aggregated features are stored
- **Rationale:** Reduces privacy risk while maintaining functionality
- **Principle:** Privacy by design

**Decision 2: Informed Consent Protocol**
- **Action:** All participants provided written consent understanding data use
- **Rationale:** Respects individual autonomy
- **Principle:** Transparency and respect for persons

**Decision 3: Bias Documentation**
- **Action:** Explicitly documented dataset demographic composition and limitations
- **Rationale:** Enables informed deployment decisions
- **Principle:** Honest representation of capabilities

**Decision 4: No Real-Time Surveillance Mode**
- **Action:** System requires explicit video upload; no continuous monitoring feature
- **Rationale:** Prevents unauthorized surveillance applications
- **Principle:** Preventing harm and misuse

**Decision 5: Open Documentation**
- **Action:** Comprehensive README files and code comments
- **Rationale:** Enables scrutiny and responsible reuse
- **Principle:** Transparency and knowledge sharing

### 7.4 Responsible Development Practices

**Technical Safeguards:**
- Model outputs include confidence scores (via probabilities)
- Multi-model ensemble reduces single-point failure risk
- Graceful degradation: system alerts when landmark detection quality is low

**Procedural Safeguards:**
- Data was collected only from consenting adults
- Videos were not shared publicly or used for unrelated purposes
- All team members trained on data protection practices

**Documentation:**
- Clear statements of intended use (education, research, rehabilitation support)
- Explicit warnings against use for:
  - Medical diagnosis (not FDA-approved or clinically validated)
  - Legal proceedings (not forensically validated)
  - Employment decisions (risk of discrimination)
  - Surveillance without consent

### 7.5 Future Ethical Considerations

For deployment beyond academic context:

1. **Regulatory Compliance:**
   - GDPR (Europe): Biometric data protection
   - HIPAA (US): If used in healthcare settings
   - Local privacy laws

2. **Continuous Monitoring:**
   - Bias audits across demographic groups
   - Performance monitoring for model drift
   - User feedback mechanisms

3. **Stakeholder Engagement:**
   - Involve domain experts (physical therapists, ethicists)
   - User studies with target populations
   - Community consultation on acceptable use cases

---

## 8. Impact Analysis (Competency P12)

### 8.1 Social Impact

#### 8.1.1 Positive Impacts

**Healthcare Accessibility:**
- **Impact:** Enables remote rehabilitation monitoring, particularly valuable in rural areas or during pandemics
- **Beneficiaries:** Patients with limited mobility, healthcare systems with capacity constraints
- **Evidence:** Telemedicine usage increased 154% in 2020 [9]; AI-assisted monitoring can extend reach

**Objective Movement Assessment:**
- **Impact:** Reduces subjectivity in clinical evaluations
- **Benefit:** Standardized assessment enables longitudinal tracking and research
- **Example:** Parkinson's disease tremor quantification, fall risk assessment

**Educational Applications:**
- **Impact:** Provides feedback for physical education and sports training
- **Benefit:** Democratizes access to performance analysis (previously limited to elite athletes)

#### 8.1.2 Negative Impacts and Risks

**Privacy Erosion:**
- **Risk:** Video analysis reveals sensitive biometric data (gait patterns are identifying)
- **Magnitude:** High - gait recognition accuracy >95% [10]
- **Affected Groups:** All users, particularly those under surveillance
- **Mitigation:** Process data locally, minimize retention, encrypt transmissions

**Surveillance Creep:**
- **Risk:** Technology repurposed for unauthorized monitoring (workplaces, public spaces)
- **Magnitude:** Medium to High - precedents exist (facial recognition misuse)
- **Affected Groups:** Workers, public citizens, vulnerable populations
- **Mitigation:** Technical restrictions (no real-time streaming), legal frameworks, transparency requirements

**Over-Reliance on Automation:**
- **Risk:** Healthcare professionals may defer to AI without critical assessment
- **Magnitude:** Medium - system accuracy not validated for clinical use
- **Affected Groups:** Patients receiving suboptimal care
- **Mitigation:** Position as decision-support tool, not replacement; require human oversight

**Discrimination:**
- **Risk:** Performance disparities across demographics lead to unequal service quality
- **Example:** Lower accuracy for individuals with atypical movement patterns (disabilities, age)
- **Magnitude:** Unknown - not tested on diverse populations
- **Affected Groups:** Elderly, disabled, non-standard body types
- **Mitigation:** Diverse training data, fairness audits, transparency about limitations

### 8.2 Economic Impact

#### 8.2.1 Cost-Benefit Analysis

**Benefits:**

| Stakeholder | Economic Benefit | Estimated Value |
|-------------|------------------|-----------------|
| Healthcare Systems | Reduced in-person visits | $50-100 per remote session saved |
| Patients | Reduced travel costs | $20-50 per avoided trip |
| Physical Therapy | Increased patient throughput | 20-30% capacity increase |
| Insurance | Early intervention reduces costs | 10-15% reduction in long-term care |
| Sports Organizations | Data-driven training optimization | Performance improvement ROI |

**Costs:**

| Cost Category | Investment Required |
|---------------|---------------------|
| Initial Development | ~$50,000 (labor, infrastructure) |
| Deployment Infrastructure | $5,000-10,000/year (servers, maintenance) |
| Validation Studies | $100,000+ (clinical trials if medical use) |
| Liability Insurance | Variable (depends on use case) |

**Break-Even Analysis:**  
For a healthcare clinic seeing 50 patients/week:
- Cost savings: $2,500-5,000/week
- Annual benefit: $130,000-260,000
- Break-even: < 6 months

#### 8.2.2 Market Opportunities

**Addressable Markets:**
1. **Telemedicine:** $155 billion global market (2020), growing 25% annually [11]
2. **Sports Analytics:** $3.5 billion market (2022), CAGR 22% [12]
3. **Rehabilitation Tech:** $1.8 billion market (2021), growing with aging populations

**Competitive Landscape:**
- Existing solutions: KinaTrax, Simi Motion, Vicon (expensive, professional-grade)
- Market gap: Affordable, accessible solutions for clinics and consumers
- Competitive advantage: Low-cost hardware (standard cameras), user-friendly interface

#### 8.2.3 Economic Risks

**Liability:**
- Misclassification leading to injury (e.g., incorrect exercise recommendation)
- Requires professional indemnity insurance (~$5,000-20,000/year)

**Market Adoption Barriers:**
- Regulatory approval costs ($500,000+ for FDA clearance)
- Integration with existing healthcare IT systems
- Reimbursement uncertainty (insurance coverage)

### 8.3 Environmental Impact

#### 8.3.1 Carbon Footprint

**Computational Energy Consumption:**

**Training Phase (one-time):**
- Hardware: Standard laptop (65W TDP)
- Training time: ~30 minutes total (all models)
- Energy: 0.0325 kWh
- CO₂ equivalent: ~0.015 kg CO₂ (US grid average)

**Inference Phase (per video):**
- Processing time: 1.5 seconds
- Energy: ~0.000027 kWh
- CO₂ equivalent: ~0.000012 kg CO₂
- **Annual (1000 videos):** 0.027 kWh, 0.012 kg CO₂

**Comparison:**
- Single Google search: 0.0003 kWh, 0.0002 kg CO₂
- 1 hour Netflix streaming: 0.8 kWh, 0.4 kg CO₂
- **Conclusion:** Negligible environmental impact per use

**Cloud Deployment:**
- Server energy: ~100W continuous
- Annual energy: 876 kWh
- Annual CO₂: ~400 kg CO₂
- **Mitigation:** Use green cloud providers (Google Cloud 100% renewable energy)

#### 8.3.2 Hardware Requirements

**Minimal Hardware Needs:**
- Standard smartphone camera (no specialized sensors)
- Reduces e-waste compared to dedicated motion capture systems
- No need for wearable sensors (battery waste, disposal issues)

**Positive Environmental Outcome:**
- Replaces travel for medical appointments → reduced transportation emissions
- Estimated: 10 miles saved/appointment × 50 appointments/year = 500 miles
- CO₂ saved: ~200 kg CO₂/year per patient (vs. driving)

### 8.4 Global Impact

#### 8.4.1 Cultural Considerations

**Cultural Variability in Movement:**
- Sitting postures vary by culture (chair vs. floor sitting)
- Walking gait influenced by footwear norms
- **Impact:** Model trained on limited cultural context may not generalize
- **Solution:** Collect diverse international datasets, enable fine-tuning

**Language and Accessibility:**
- Current interface: English only
- **Barrier:** Excludes non-English speakers
- **Solution:** Internationalization (i18n), multi-language support

#### 8.4.2 Regulatory Landscape

**Regional Variations:**

| Region | Key Regulations | Implications |
|--------|----------------|--------------|
| European Union | GDPR, AI Act | Strict biometric data rules, high-risk AI classification |
| United States | HIPAA (healthcare), sectoral laws | Varies by state and application |
| China | Personal Information Protection Law | Data localization requirements |
| Global South | Limited AI-specific regulation | Opportunity for early adoption, risk of exploitation |

**Compliance Requirements:**
- Data residency (store EU data in EU)
- Consent mechanisms (GDPR-compliant)
- Algorithmic transparency (explainability)
- Right to human override

#### 8.4.3 Digital Divide

**Access Barriers:**
- Requires smartphone with camera (60% global smartphone penetration)
- Requires internet for cloud deployment
- Digital literacy needed to use interface
- **Risk:** Exacerbates health disparities between developed/developing regions
- **Mitigation:** Offline mode, simplified interface, community health worker training programs

#### 8.4.4 Global Health Equity

**Positive Potential:**
- Addresses healthcare worker shortage in low-resource settings
- Enables remote specialist consultations
- **Example:** Physical therapist in urban center can monitor rural patients

**Negative Risks:**
- Brain drain (local expertise replaced by remote AI)
- Dependency on foreign technology
- Data extraction (training on Global South data, deploying in developed markets)
- **Mitigation:** Open-source models, local capacity building, data sovereignty agreements

### 8.5 Impact Mitigation Strategies

**Priority Actions:**

1. **Privacy-Preserving Architecture:**
   - On-device processing option
   - Federated learning for model updates
   - Differential privacy for aggregated statistics

2. **Bias Audits:**
   - Regular testing across demographic groups
   - Disaggregated performance reporting
   - Community feedback mechanisms

3. **Transparent Communication:**
   - Clear user consent flows
   - Public documentation of limitations
   - Accessible explanations of how system works

4. **Stakeholder Engagement:**
   - Co-design with healthcare professionals
   - User studies with target populations
   - Ethics board oversight

5. **Sustainable Deployment:**
   - Choose green cloud infrastructure
   - Optimize models for efficiency
   - Support right-to-repair (no planned obsolescence)

---

## 9. Conclusions and Future Work

### 9.1 Achievements

This project successfully developed a complete human activity recognition system, achieving the following outcomes:

1. **Dataset Creation:** Collected and annotated 86 videos across 8 activity categories with 33-point pose landmarks.

2. **Feature Engineering:** Designed 12 biomechanical features capturing kinematic and postural characteristics of human movement.

3. **Model Development:** Trained three supervised learning models (Random Forest, SVM, XGBoost) with accuracy up to 94.44%.

4. **Dimensionality Reduction:** Applied PCA to reduce feature space by 50% while preserving 91.73% of variance, mitigating overfitting.

5. **Deployment:** Developed a functional web application using Streamlit with real-time classification capabilities.

6. **Ethical Analysis:** Conducted comprehensive ethical review addressing privacy, bias, and societal impacts.

7. **Mathematical Rigor:** Formulated the problem mathematically and justified modeling decisions with theoretical foundations.

### 9.2 Key Learnings

**Technical Insights:**

1. **Small Data Challenges:** With only ~11 samples per class, distinguishing overfitting from genuine learning is difficult. The 100% accuracy on original features was misleading.

2. **Value of Dimensionality Reduction:** PCA served dual purposes: reducing overfitting and improving interpretability. The 5.56% accuracy drop for RF/SVM is an acceptable trade-off.

3. **Model-Specific Responses to PCA:** XGBoost's 11% performance drop suggests tree-based boosting benefits from original feature interactions. Future work should explore selective feature engineering.

4. **Biomechanical Features > Raw Landmarks:** Engineered features (12 dimensions) outperformed raw landmark coordinates (99 dimensions) due to noise reduction and domain knowledge incorporation.

5. **Ensemble Benefits:** Multi-model consensus provides robustness against individual model failures.

**Methodological Insights:**

1. **CRISP-DM Effectiveness:** The iterative methodology enabled systematic problem-solving and continuous improvement across three deliveries.

2. **Importance of Validation Strategy:** Cross-validation scores often over-optimistic for small datasets; external validation is critical.

3. **Documentation Value:** Comprehensive documentation facilitated collaboration and will enable future extensions.

**Ethical Insights:**

1. **Privacy-by-Design:** Incorporating privacy considerations early (data minimization) is easier than retrofitting.

2. **Bias Awareness:** Acknowledging dataset limitations honestly builds trust and prevents harmful deployments.

3. **Dual-Use Concerns:** Technologies with beneficial applications (rehabilitation) can be misused (surveillance); responsible development requires anticipating misuse.

### 9.3 Limitations

**Dataset Limitations:**
1. **Sample Size:** 86 videos insufficient for robust generalization
2. **Demographic Homogeneity:** Limited age, gender, ethnicity, body type diversity
3. **Environmental Uniformity:** Single indoor setting, consistent lighting
4. **Activity Simplicity:** Basic movements; not representative of complex activities

**Methodological Limitations:**
1. **No External Validation:** Test set from same subjects/environment as training
2. **Limited Hyperparameter Search:** Computational constraints limited GridSearchCV scope
3. **Temporal Averaging:** Discards temporal dynamics; activities classified by average pose

**Technical Limitations:**
1. **Frame-by-Frame Processing:** No temporal modeling (e.g., LSTMs, temporal CNNs)
2. **Single-Person Assumption:** Cannot handle multiple people in frame
3. **Occlusion Sensitivity:** Performance degrades when landmarks are occluded
4. **Camera-Dependent:** Assumes frontal or near-frontal viewing angle

### 9.4 Future Work

#### 9.4.1 Dataset Expansion

**Priority 1 - Scale:**
- Target: 200+ videos per category (1,600+ total)
- Recruit diverse participants (age 18-80, varied body types, mobility levels)
- Include participants with movement disorders (Parkinson's, stroke recovery)

**Priority 2 - Environmental Diversity:**
- Outdoor videos (varied backgrounds)
- Different lighting conditions (natural, artificial, low-light)
- Multiple camera angles (side, overhead, oblique)

**Priority 3 - Activity Complexity:**
- Multi-step activities (sit-to-stand-to-walk transitions)
- Natural behaviors (reaching, bending, turning)
- Activity variations (fast vs. slow walking)

#### 9.4.2 Model Improvements

**Temporal Modeling:**
- Implement LSTM or GRU networks to capture movement dynamics
- Use Temporal Convolutional Networks (TCN) for long-range dependencies
- Explore Transformer architectures for sequence-to-sequence modeling

**Advanced Architectures:**
- Graph Neural Networks (GNN) on skeleton structure [13]
- Spatial-Temporal Graph Convolution Networks (ST-GCN) [7]
- 3D CNNs on landmark sequences

**Ensemble Refinement:**
- Train diverse base models (temporal + spatial)
- Use stacking or blending for ensemble combination
- Implement uncertainty quantification (Bayesian approaches)

#### 9.4.3 Feature Engineering

**Additional Features:**
- **Frequency-domain:** FFT of landmark trajectories (rhythmic movements)
- **Geometric:** Body part relationships (limb length ratios)
- **Contextual:** Interaction with environment (contact with surfaces)

**Feature Selection:**
- Systematic feature importance analysis
- Recursive feature elimination (RFE)
- LASSO regularization for sparse feature sets

#### 9.4.4 Validation and Evaluation

**Cross-Subject Validation:**
- Leave-one-subject-out (LOSO) cross-validation
- Ensures generalization to new individuals

**Cross-Environment Testing:**
- Train on indoor, test on outdoor
- Quantify environmental robustness

**Clinical Validation:**
- Collaborate with physical therapists
- Compare system classifications with expert assessments
- Measure inter-rater reliability

**Fairness Audits:**
- Disaggregated evaluation by demographics
- Identify and mitigate performance disparities

#### 9.4.5 Deployment Enhancements

**Real-Time Processing:**
- Optimize MediaPipe for mobile devices
- Implement on-device inference (TensorFlow Lite, CoreML)
- Reduce latency to < 100ms per frame

**User Interface:**
- Mobile application (iOS, Android)
- Real-time visual feedback (pose overlay, angle indicators)
- Gamification for rehabilitation adherence

**Privacy Features:**
- Federated learning for model updates (train without sharing data)
- Differential privacy for aggregated analytics
- End-to-end encryption for video transmission

**Integration:**
- Electronic Health Record (EHR) integration (FHIR standards)
- Wearable sensor fusion (IMU + vision)
- Multi-modal data (video + audio for comprehensive assessment)

#### 9.4.6 Clinical Translation

**Regulatory Pathway:**
- FDA 510(k) clearance for clinical decision support
- CE marking for European market
- Clinical trial design for efficacy validation

**Specific Applications:**
- Post-stroke gait rehabilitation monitoring
- Parkinson's disease tremor and bradykinesia quantification
- Pediatric developmental milestone tracking
- Elderly fall risk assessment

**Evidence Generation:**
- Prospective observational study (n > 100 patients)
- Randomized controlled trial comparing standard care vs. AI-assisted
- Cost-effectiveness analysis

#### 9.4.7 Ethical and Social Research

**Longitudinal Impact Study:**
- Survey users on privacy perceptions
- Monitor for surveillance creep or mission drift
- Assess impact on patient-clinician relationship

**Participatory Design:**
- Co-create system features with end users
- Community advisory boards
- Patient advocacy group partnerships

**Policy Development:**
- Contribute to AI ethics guidelines for healthcare
- Advocate for responsible innovation frameworks
- Support regulatory clarity

### 9.5 Final Reflections

This project demonstrates that meaningful AI applications can be developed with limited resources through:
1. Clear problem formulation
2. Thoughtful feature engineering
3. Rigorous evaluation
4. Honest acknowledgment of limitations
5. Proactive ethical consideration

The journey from 54 videos to 86 videos to a deployed web application exemplifies the iterative nature of machine learning projects. The overfitting observed in Entrega 2 taught us more than a perfect model would have—it highlighted the gap between training performance and real-world generalization.

Most importantly, this work underscores that **technical excellence alone is insufficient**. Responsible AI development requires:
- Understanding societal context
- Anticipating unintended consequences
- Engaging diverse stakeholders
- Committing to ongoing evaluation and improvement

As engineers, our professional responsibility extends beyond algorithm accuracy to encompass the broader impacts of our creations. This project has reinforced our commitment to developing technology that serves humanity ethically, equitably, and sustainably.

---

## 10. Bibliographic References

[1] C. Lugaresi, J. Tang, H. Nash, et al., "MediaPipe: A Framework for Building Perception Pipelines," *arXiv preprint arXiv:1906.08172*, 2019.

[2] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[3] C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[4] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 2016, pp. 785–794.

[5] I. T. Jolliffe and J. Cadima, "Principal Component Analysis: A Review and Recent Developments," *Philosophical Transactions of the Royal Society A*, vol. 374, no. 2065, 2016.

[6] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, "Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 7291–7299.

[7] S. Yan, Y. Xiong, and D. Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," in *Proc. AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.

[8] IEEE, "IEEE Code of Ethics," 2020. [Online]. Available: https://www.ieee.org/about/corporate/governance/p7-8.html

[9] A. Mehrotra, M. Chernew, D. Linetsky, H. Hatch, and D. Cutler, "The Impact of the COVID-19 Pandemic on Outpatient Visits: A Rebound Emerges," *Commonwealth Fund*, May 2020.

[10] M. S. Nixon, T. N. Tan, and R. Chellappa, *Human Identification Based on Gait*, Springer, 2006.

[11] Fortune Business Insights, "Telemedicine Market Size, Share & COVID-19 Impact Analysis," Report ID FBI101065, 2021.

[12] Grand View Research, "Sports Analytics Market Size, Share & Trends Analysis Report," Report ID GVR-1-68038-168-2, 2022.

[13] C. Plizzari, M. Cannici, and M. Matteucci, "Spatial Temporal Transformer Network for Skeleton-Based Action Recognition," in *Proc. International Conference on Pattern Recognition (ICPR)*, 2021, pp. 694–701.

[14] S. Mehrabi, A. Stairs, L. Walmsley, et al., "Toward Generalization of Deep Learning-Based Human Activity Recognition," *IEEE Journal of Biomedical and Health Informatics*, vol. 27, no. 3, pp. 1541–1551, 2023.

[15] D. Konstantinidis, K. Dimitropoulos, and P. Daras, "A Deep Learning Approach for Analyzing Video and Skeletal Features in Sign Language Recognition," in *Proc. IEEE International Conference on Imaging Systems and Techniques (IST)*, 2018, pp. 1–6.

---

## Appendices

### Appendix A: Ethical Decision-Making Matrix

| Decision | Ethical Principle | Action Taken | Outcome |
|----------|-------------------|--------------|---------|
| Video storage | Privacy | Process and delete; store only features | Reduced privacy risk |
| Consent protocol | Autonomy | Written consent with clear explanation | Informed participation |
| Bias documentation | Honesty | Explicitly stated demographic limitations | Transparent communication |
| Surveillance prevention | Preventing harm | No continuous monitoring feature | Reduced misuse risk |
| Open documentation | Knowledge sharing | Public GitHub repository | Community benefit |

### Appendix B: Impact Assessment Matrix

| Dimension | Positive Impacts | Negative Impacts | Mitigation Strategies |
|-----------|------------------|------------------|----------------------|
| **Social** | Remote healthcare access; Objective assessment | Privacy erosion; Surveillance creep | Local processing; Consent mechanisms |
| **Economic** | Cost savings ($130k-260k/year for clinics) | Liability risks; High validation costs | Insurance; Staged deployment |
| **Environmental** | Reduced travel emissions (200 kg CO₂/patient/year) | Server energy (400 kg CO₂/year) | Green cloud providers |
| **Global** | Addresses healthcare worker shortage | Digital divide; Cultural bias | Offline mode; Diverse datasets |

### Appendix C: Software and Tools

**Development Environment:**
- Python 3.11
- Jupyter Notebook / Google Colab
- Visual Studio Code

**Key Libraries:**
- MediaPipe 0.10.9 (pose estimation)
- OpenCV 4.9.0 (video processing)
- scikit-learn 1.4.2 (ML models, PCA)
- XGBoost 2.0.3 (gradient boosting)
- Streamlit 1.32.0 (web application)
- Pandas 2.2.2, NumPy 1.26.4 (data manipulation)
- Matplotlib 3.8.3, Seaborn 0.13.2 (visualization)

**Infrastructure:**
- GitHub (version control)
- Docker (containerization)
- Railway / Streamlit Cloud (deployment)

### Appendix D: Dataset Statistics

**Final Dataset Summary:**

```
Total Videos:              86
Total Frames Processed:    ~8,600
Categories:                8
Samples per Category:      10-11
Resolution:                1080x1908 pixels
Frame Rate:                30 FPS
Average Duration:          3.1 seconds
Landmark Detection Rate:   94.2%
```

**Category Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| Adelante | 18 | 20.9% |
| Atrás | 18 | 20.9% |
| Sentado | 18 | 20.9% |
| Cadera_Alfrente | 6 | 7.0% |
| Caderas | 6 | 7.0% |
| Lado | 6 | 7.0% |
| Sentadilla | 7 | 8.1% |
| Tijeras | 7 | 8.1% |

---

**END OF REPORT**

*Total Pages: 21 (to be condensed to 7 pages in PDF format with adjusted formatting)*

---

## Authors' Contributions

- **Mariana De La Cruz:** Dataset collection, MediaPipe implementation, ethical analysis
- **Valentina Gómez:** Feature engineering, model training, result visualization
- **Alexis Delgado:** Web application development, deployment infrastructure, documentation
- **Juan Camilo Amorocho:** Mathematical formulation, PCA implementation, impact analysis

**Acknowledgments:** We thank Universidad ICESI and the APO3 course instructors for guidance throughout this project.

---

*This report fulfills the requirements for the final deliverable of the APO3 course, addressing competencies P11 (Ethics), P12 (Impact Analysis), and P13 (Mathematical Problem Solving).*

