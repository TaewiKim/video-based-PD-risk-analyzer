"""
CARE-PD PKL 데이터셋 시각화 스크립트
- 데이터셋 요약 통계
- UPDRS Gait Score 분포
- 시퀀스 길이 분포
- 3D 보행 궤적 (translation)
- SMPL 관절 각도 시계열
- 스틱 피겨 시각화
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── 설정 ──
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets" / "CARE-PD"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "pkl_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILES = [
    "3DGait.pkl", "BMCLab.pkl", "DNE.pkl", "E-LC.pkl",
    "KUL-DT-T.pkl", "PD-GaM.pkl", "T-LTC.pkl", "T-SDU-PD.pkl", "T-SDU.pkl"
]

# SMPL 24 joint names
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

# SMPL skeleton connections (parent-child)
SMPL_SKELETON = [
    (0, 1), (0, 2), (0, 3),       # Pelvis -> L_Hip, R_Hip, Spine1
    (1, 4), (2, 5),               # Hips -> Knees
    (3, 6),                        # Spine1 -> Spine2
    (4, 7), (5, 8),               # Knees -> Ankles
    (6, 9),                        # Spine2 -> Spine3
    (7, 10), (8, 11),             # Ankles -> Feet
    (9, 12), (9, 13), (9, 14),    # Spine3 -> Neck, L/R_Collar
    (12, 15),                      # Neck -> Head
    (13, 16), (14, 17),           # Collars -> Shoulders
    (16, 18), (17, 19),           # Shoulders -> Elbows
    (18, 20), (19, 21),           # Elbows -> Wrists
    (20, 22), (21, 23),           # Wrists -> Hands
]


def load_all_datasets():
    """모든 데이터셋 로드"""
    datasets = {}
    for fname in DATASET_FILES:
        path = DATA_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                datasets[fname.replace(".pkl", "")] = pickle.load(f)
            print(f"  Loaded {fname}")
        else:
            print(f"  [SKIP] {fname} not found")
    return datasets


def collect_stats(datasets):
    """데이터셋별 통계 수집"""
    stats = {}
    for ds_name, data in datasets.items():
        n_subjects = len(data)
        seq_lengths = []
        updrs_scores = []
        medications = []
        fps_list = []

        for subj_key, trials in data.items():
            for trial_key, trial in trials.items():
                seq_lengths.append(trial["pose"].shape[0])
                fps_list.append(trial["fps"])
                if trial.get("UPDRS_GAIT") is not None:
                    updrs_scores.append(trial["UPDRS_GAIT"])
                if trial.get("medication") is not None:
                    medications.append(trial["medication"])

        stats[ds_name] = {
            "n_subjects": n_subjects,
            "n_trials": len(seq_lengths),
            "seq_lengths": seq_lengths,
            "updrs_scores": updrs_scores,
            "medications": medications,
            "fps": fps_list[0] if fps_list else 0,
            "total_frames": sum(seq_lengths),
        }
    return stats


def axis_angle_to_rotation_matrix(axis_angle):
    """Axis-angle (3,) -> Rotation matrix (3,3)"""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def forward_kinematics(pose, trans=None):
    """SMPL pose parameters -> 3D joint positions via forward kinematics

    Simplified FK using T-pose offsets (approximate but sufficient for visualization).
    """
    # Approximate T-pose offsets (relative to parent joint)
    t_pose_offsets = np.array([
        [0.0, 0.0, 0.0],        # 0  Pelvis
        [0.07, -0.05, 0.0],     # 1  L_Hip
        [-0.07, -0.05, 0.0],    # 2  R_Hip
        [0.0, 0.1, 0.0],        # 3  Spine1
        [0.0, -0.45, 0.0],      # 4  L_Knee
        [0.0, -0.45, 0.0],      # 5  R_Knee
        [0.0, 0.15, 0.0],       # 6  Spine2
        [0.0, -0.45, 0.0],      # 7  L_Ankle
        [0.0, -0.45, 0.0],      # 8  R_Ankle
        [0.0, 0.15, 0.0],       # 9  Spine3
        [0.0, -0.05, 0.08],     # 10 L_Foot
        [0.0, -0.05, 0.08],     # 11 R_Foot
        [0.0, 0.12, 0.0],       # 12 Neck
        [0.08, 0.05, 0.0],      # 13 L_Collar
        [-0.08, 0.05, 0.0],     # 14 R_Collar
        [0.0, 0.12, 0.0],       # 15 Head
        [0.12, 0.0, 0.0],       # 16 L_Shoulder
        [-0.12, 0.0, 0.0],      # 17 R_Shoulder
        [0.25, 0.0, 0.0],       # 18 L_Elbow
        [-0.25, 0.0, 0.0],      # 19 R_Elbow
        [0.25, 0.0, 0.0],       # 20 L_Wrist
        [-0.25, 0.0, 0.0],      # 21 R_Wrist
        [0.08, 0.0, 0.0],       # 22 L_Hand
        [-0.08, 0.0, 0.0],      # 23 R_Hand
    ])

    parent_indices = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    n_frames = pose.shape[0]
    joints_3d = np.zeros((n_frames, 24, 3))

    for frame_idx in range(n_frames):
        pose_frame = pose[frame_idx].reshape(24, 3)
        global_rotations = [None] * 24
        global_positions = [None] * 24

        for j in range(24):
            local_rot = axis_angle_to_rotation_matrix(pose_frame[j])
            parent = parent_indices[j]
            if parent == -1:
                global_rotations[j] = local_rot
                global_positions[j] = trans[frame_idx] if trans is not None else np.zeros(3)
            else:
                global_rotations[j] = global_rotations[parent] @ local_rot
                global_positions[j] = global_positions[parent] + global_rotations[parent] @ t_pose_offsets[j]

        for j in range(24):
            joints_3d[frame_idx, j] = global_positions[j]

    return joints_3d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 1: 데이터셋 요약 테이블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_summary_table(stats):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    headers = ["Dataset", "Subjects", "Trials", "FPS", "Frames\n(total)",
               "Seq Len\n(mean)", "Seq Len\n(std)", "UPDRS\nAvailable"]
    rows = []
    for ds_name in sorted(stats.keys()):
        s = stats[ds_name]
        sl = s["seq_lengths"]
        rows.append([
            ds_name, s["n_subjects"], s["n_trials"], s["fps"],
            f"{s['total_frames']:,}",
            f"{np.mean(sl):.0f}", f"{np.std(sl):.0f}",
            f"Yes ({len(s['updrs_scores'])})" if s["updrs_scores"] else "No"
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    ax.set_title("CARE-PD Dataset Summary", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_dataset_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [1/6] Dataset summary saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 2: 시퀀스 길이 분포
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_sequence_lengths(stats):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, ds_name in enumerate(sorted(stats.keys())):
        ax = axes[idx]
        sl = stats[ds_name]["seq_lengths"]
        ax.hist(sl, bins=30, color="#4472C4", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(sl), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(sl):.0f}")
        ax.set_title(ds_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Sequence Length Distribution per Dataset", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_sequence_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/6] Sequence length distribution saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 3: UPDRS Gait Score 분포
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_updrs_distribution(stats):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Collect all UPDRS scores with dataset label
    updrs_datasets = {ds: s["updrs_scores"] for ds, s in stats.items() if s["updrs_scores"]}

    if not updrs_datasets:
        plt.close(fig)
        print("  [3/6] No UPDRS data found, skipping")
        return

    # Plot 1: Combined distribution
    all_scores = []
    for scores in updrs_datasets.values():
        all_scores.extend(scores)

    axes[0].hist(all_scores, bins=range(0, max(all_scores) + 2), color="#4472C4",
                 alpha=0.8, edgecolor="white", align="left")
    axes[0].set_title("All Datasets Combined", fontweight="bold")
    axes[0].set_xlabel("UPDRS Gait Score")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(range(0, max(all_scores) + 1))

    # Plot 2-4: Per-dataset
    for idx, (ds_name, scores) in enumerate(sorted(updrs_datasets.items())):
        if idx >= 3:
            break
        ax = axes[idx + 1]
        ax.hist(scores, bins=range(0, max(scores) + 2), color=plt.cm.Set2(idx),
                alpha=0.8, edgecolor="white", align="left")
        ax.set_title(ds_name, fontweight="bold")
        ax.set_xlabel("UPDRS Gait Score")
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, max(scores) + 1))

    fig.suptitle("UPDRS Gait Score Distribution", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_updrs_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3/6] UPDRS distribution saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 4: 3D 보행 궤적 (Translation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_walking_trajectories(datasets):
    fig = plt.figure(figsize=(18, 12))

    for idx, ds_name in enumerate(sorted(datasets.keys())):
        ax = fig.add_subplot(3, 3, idx + 1, projection="3d")
        data = datasets[ds_name]

        # Plot up to 10 random subjects
        subj_keys = list(data.keys())[:10]
        colors = plt.cm.tab10(np.linspace(0, 1, len(subj_keys)))

        for s_idx, subj_key in enumerate(subj_keys):
            trials = data[subj_key]
            first_trial_key = list(trials.keys())[0]
            trial = trials[first_trial_key]
            trans = trial["trans"]

            ax.plot(trans[:, 0], trans[:, 2], trans[:, 1],
                    alpha=0.7, linewidth=0.8, color=colors[s_idx])
            ax.scatter(trans[0, 0], trans[0, 2], trans[0, 1],
                       c="green", s=15, zorder=5)
            ax.scatter(trans[-1, 0], trans[-1, 2], trans[-1, 1],
                       c="red", s=15, zorder=5)

        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Z", fontsize=8)
        ax.set_zlabel("Y", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle("Walking Trajectories (Translation) - 1st trial per subject\n"
                 "(Green=start, Red=end)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_walking_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/6] Walking trajectories saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 5: 주요 관절 각도 시계열
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_joint_angles(datasets):
    key_joints = [1, 2, 4, 5, 7, 8]  # L/R Hip, Knee, Ankle
    joint_labels = ["L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Pick a sample dataset and subject
    sample_ds_name = "BMCLab" if "BMCLab" in datasets else list(datasets.keys())[0]
    data = datasets[sample_ds_name]
    first_subj = list(data.keys())[0]
    first_trial_key = list(data[first_subj].keys())[0]
    trial = data[first_subj][first_trial_key]
    pose = trial["pose"]
    fps = trial["fps"]

    n_frames = pose.shape[0]
    time = np.arange(n_frames) / fps

    for idx, (joint_idx, joint_name) in enumerate(zip(key_joints, joint_labels)):
        ax = axes[idx]
        # Extract 3 axis-angle components for this joint
        joint_pose = pose[:, joint_idx * 3: joint_idx * 3 + 3]
        angles_deg = np.rad2deg(np.linalg.norm(joint_pose, axis=1))

        ax.plot(time, joint_pose[:, 0], alpha=0.8, label="X (Flexion)", linewidth=0.8)
        ax.plot(time, joint_pose[:, 1], alpha=0.8, label="Y (Abduction)", linewidth=0.8)
        ax.plot(time, joint_pose[:, 2], alpha=0.8, label="Z (Rotation)", linewidth=0.8)
        ax.set_title(f"{joint_name} (Joint {joint_idx})", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Axis-Angle (rad)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Joint Angle Time Series\n{sample_ds_name} / {first_subj} / {first_trial_key}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_joint_angles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [5/6] Joint angle time series saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 6: 스틱 피겨 시각화 (여러 프레임)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_stick_figures(datasets):
    # Pick a sample with reasonable length
    sample_ds_name = "BMCLab" if "BMCLab" in datasets else list(datasets.keys())[0]
    data = datasets[sample_ds_name]
    first_subj = list(data.keys())[0]
    first_trial_key = list(data[first_subj].keys())[0]
    trial = data[first_subj][first_trial_key]

    pose = trial["pose"]
    trans = trial["trans"]
    n_frames = pose.shape[0]

    # Compute 3D joints via forward kinematics
    print("    Computing forward kinematics...")
    joints_3d = forward_kinematics(pose, trans)

    # Select 8 evenly spaced frames
    frame_indices = np.linspace(0, n_frames - 1, 8, dtype=int)

    fig = plt.figure(figsize=(20, 6))

    for plot_idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(1, 8, plot_idx + 1, projection="3d")
        joints = joints_3d[frame_idx]

        # Draw bones
        for start, end in SMPL_SKELETON:
            xs = [joints[start, 0], joints[end, 0]]
            ys = [joints[start, 2], joints[end, 2]]
            zs = [joints[start, 1], joints[end, 1]]
            ax.plot(xs, ys, zs, c="#4472C4", linewidth=2)

        # Draw joints
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
                   c="red", s=10, zorder=5)

        ax.set_title(f"F{frame_idx}", fontsize=9)
        ax.set_xlim(joints[:, 0].mean() - 0.8, joints[:, 0].mean() + 0.8)
        ax.set_ylim(joints[:, 2].mean() - 0.8, joints[:, 2].mean() + 0.8)
        ax.set_zlim(joints[:, 1].min() - 0.2, joints[:, 1].max() + 0.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=15, azim=-60)

    fig.suptitle(f"Stick Figure Sequence (Forward Kinematics)\n"
                 f"{sample_ds_name} / {first_subj} / {first_trial_key}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_stick_figures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [6/6] Stick figures saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 7: CARE-PD vs vida-adl_CARE-PD 비교
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_dataset_comparison():
    vida_dir = Path(__file__).resolve().parent.parent / "data" / "datasets" / "vida-adl_CARE-PD"
    if not vida_dir.exists():
        print("  [Bonus] vida-adl_CARE-PD not found, skipping comparison")
        return

    comparison = []
    for fname in DATASET_FILES:
        care_path = DATA_DIR / fname
        vida_path = vida_dir / fname
        if care_path.exists() and vida_path.exists():
            with open(care_path, "rb") as f:
                care_data = pickle.load(f)
            with open(vida_path, "rb") as f:
                vida_data = pickle.load(f)

            care_subjects = len(care_data)
            vida_subjects = len(vida_data)
            care_trials = sum(len(v) for v in care_data.values())
            vida_trials = sum(len(v) for v in vida_data.values())

            comparison.append([
                fname.replace(".pkl", ""),
                care_subjects, vida_subjects,
                care_trials, vida_trials,
                "Same" if care_subjects == vida_subjects and care_trials == vida_trials else "Different"
            ])

    if not comparison:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    headers = ["Dataset", "CARE-PD\nSubjects", "vida-adl\nSubjects",
               "CARE-PD\nTrials", "vida-adl\nTrials", "Match"]
    table = ax.table(cellText=comparison, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(comparison) + 1):
        match_val = comparison[i - 1][-1]
        color = "#C6EFCE" if match_val == "Same" else "#FFC7CE"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color if j == 5 else ("#D9E2F3" if i % 2 == 0 else "white"))

    ax.set_title("CARE-PD vs vida-adl_CARE-PD Comparison", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_dataset_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [Bonus] Dataset comparison saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("Loading datasets...")
    datasets = load_all_datasets()

    print("\nCollecting statistics...")
    stats = collect_stats(datasets)

    print("\nGenerating visualizations...")
    plot_summary_table(stats)
    plot_sequence_lengths(stats)
    plot_updrs_distribution(stats)
    plot_walking_trajectories(datasets)
    plot_joint_angles(datasets)
    plot_stick_figures(datasets)
    plot_dataset_comparison()

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")
