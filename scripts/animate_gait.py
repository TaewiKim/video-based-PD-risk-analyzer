"""
CARE-PD SMPL 보행 데이터 → 3D 스틱 피겨 애니메이션 (MP4)

사용법:
  python scripts/animate_gait.py                          # 기본 샘플 생성
  python scripts/animate_gait.py --dataset BMCLab         # 특정 데이터셋
  python scripts/animate_gait.py --dataset BMCLab --subject SUB01  # 특정 피험자
  python scripts/animate_gait.py --all                    # 각 데이터셋별 1개씩 전부 생성
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets" / "CARE-PD"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "pkl_animation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SMPL skeleton connections
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]

# Bone colors: left=blue, right=red, center=gray
BONE_COLORS = [
    "#888",    # Pelvis-L_Hip
    "#888",    # Pelvis-R_Hip
    "#888",    # Pelvis-Spine1
    "#3b82f6", # L_Hip-L_Knee
    "#ef4444", # R_Hip-R_Knee
    "#888",    # Spine1-Spine2
    "#3b82f6", # L_Knee-L_Ankle
    "#ef4444", # R_Knee-R_Ankle
    "#888",    # Spine2-Spine3
    "#3b82f6", # L_Ankle-L_Foot
    "#ef4444", # R_Ankle-R_Foot
    "#888",    # Spine3-Neck
    "#3b82f6", # Spine3-L_Collar
    "#ef4444", # Spine3-R_Collar
    "#888",    # Neck-Head
    "#3b82f6", # L_Collar-L_Shoulder
    "#ef4444", # R_Collar-R_Shoulder
    "#3b82f6", # L_Shoulder-L_Elbow
    "#ef4444", # R_Shoulder-R_Elbow
    "#3b82f6", # L_Elbow-L_Wrist
    "#ef4444", # R_Elbow-R_Wrist
    "#3b82f6", # L_Wrist-L_Hand
    "#ef4444", # R_Wrist-R_Hand
]

T_POSE_OFFSETS = np.array([
    [0.0, 0.0, 0.0],
    [0.07, -0.05, 0.0],
    [-0.07, -0.05, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, -0.45, 0.0],
    [0.0, -0.45, 0.0],
    [0.0, 0.15, 0.0],
    [0.0, -0.45, 0.0],
    [0.0, -0.45, 0.0],
    [0.0, 0.15, 0.0],
    [0.0, -0.05, 0.08],
    [0.0, -0.05, 0.08],
    [0.0, 0.12, 0.0],
    [0.08, 0.05, 0.0],
    [-0.08, 0.05, 0.0],
    [0.0, 0.12, 0.0],
    [0.12, 0.0, 0.0],
    [-0.12, 0.0, 0.0],
    [0.25, 0.0, 0.0],
    [-0.25, 0.0, 0.0],
    [0.25, 0.0, 0.0],
    [-0.25, 0.0, 0.0],
    [0.08, 0.0, 0.0],
    [-0.08, 0.0, 0.0],
])

PARENT_INDICES = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def axis_angle_to_rotation_matrix(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-6:
        return np.eye(3)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def forward_kinematics_batch(pose, trans):
    """Batch FK: pose (N,72), trans (N,3) -> joints (N,24,3)"""
    n_frames = pose.shape[0]
    joints = np.zeros((n_frames, 24, 3))

    for f in range(n_frames):
        pose_f = pose[f].reshape(24, 3)
        g_rot = [None] * 24
        g_pos = [None] * 24
        for j in range(24):
            local_rot = axis_angle_to_rotation_matrix(pose_f[j])
            p = PARENT_INDICES[j]
            if p == -1:
                g_rot[j] = local_rot
                g_pos[j] = trans[f]
            else:
                g_rot[j] = g_rot[p] @ local_rot
                g_pos[j] = g_pos[p] + g_rot[p] @ T_POSE_OFFSETS[j]
        for j in range(24):
            joints[f, j] = g_pos[j]

    return joints


def create_animation(joints, fps, title, output_path, max_frames=None):
    """3D stick figure animation -> MP4"""
    n_frames = joints.shape[0]
    if max_frames and n_frames > max_frames:
        # Subsample to keep animation reasonable
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        joints = joints[indices]
        n_frames = max_frames

    # Compute global bounds
    margin = 0.3
    x_min, x_max = joints[:, :, 0].min() - margin, joints[:, :, 0].max() + margin
    y_min, y_max = joints[:, :, 1].min() - margin, joints[:, :, 1].max() + margin
    z_min, z_max = joints[:, :, 2].min() - margin, joints[:, :, 2].max() + margin

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw floor grid
    floor_y = y_min
    grid_x = np.linspace(x_min, x_max, 10)
    grid_z = np.linspace(z_min, z_max, 10)

    bone_lines = []
    for i, (start, end) in enumerate(SMPL_BONES):
        line, = ax.plot([], [], [], c=BONE_COLORS[i], linewidth=2.5, solid_capstyle="round")
        bone_lines.append(line)

    joint_scatter = ax.scatter([], [], [], c="white", edgecolors="black",
                                s=25, zorder=5, linewidths=0.8)

    # Trail line (foot trajectory)
    trail_l, = ax.plot([], [], [], c="#3b82f6", alpha=0.3, linewidth=1)
    trail_r, = ax.plot([], [], [], c="#ef4444", alpha=0.3, linewidth=1)

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
                          fontweight="bold", color="#333")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(y_min, y_max)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Z", fontsize=9)
    ax.set_zlabel("Y", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.view_init(elev=15, azim=-60)
    ax.tick_params(labelsize=7)

    # Draw faint floor grid
    for gx in grid_x:
        ax.plot([gx, gx], [z_min, z_max], [floor_y, floor_y], c="#ddd", linewidth=0.5)
    for gz in grid_z:
        ax.plot([x_min, x_max], [gz, gz], [floor_y, floor_y], c="#ddd", linewidth=0.5)

    def update(frame):
        j = joints[frame]
        for i, (s, e) in enumerate(SMPL_BONES):
            bone_lines[i].set_data_3d(
                [j[s, 0], j[e, 0]],
                [j[s, 2], j[e, 2]],
                [j[s, 1], j[e, 1]]
            )
        joint_scatter._offsets3d = (j[:, 0], j[:, 2], j[:, 1])

        # Foot trails (last 30 frames)
        trail_start = max(0, frame - 30)
        trail_l.set_data_3d(
            joints[trail_start:frame+1, 7, 0],
            joints[trail_start:frame+1, 7, 2],
            joints[trail_start:frame+1, 7, 1]
        )
        trail_r.set_data_3d(
            joints[trail_start:frame+1, 8, 0],
            joints[trail_start:frame+1, 8, 2],
            joints[trail_start:frame+1, 8, 1]
        )

        t = frame / fps
        time_text.set_text(f"Frame {frame}/{n_frames}  |  {t:.2f}s")
        return bone_lines + [joint_scatter, trail_l, trail_r, time_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    writer = FFMpegWriter(fps=min(fps, 30), bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path.name} ({n_frames} frames)")


def process_trial(dataset_name, data, subject_key, trial_key, max_frames=600):
    """단일 trial 처리 및 애니메이션 생성"""
    trial = data[subject_key][trial_key]
    pose = trial["pose"]
    trans = trial["trans"]
    fps = trial["fps"]
    updrs = trial.get("UPDRS_GAIT")
    med = trial.get("medication")

    print(f"  Computing FK for {dataset_name}/{subject_key}/{trial_key} "
          f"({pose.shape[0]} frames @ {fps}fps)...")
    joints = forward_kinematics_batch(pose, trans)

    title_parts = [f"{dataset_name} / {subject_key} / {trial_key}"]
    if updrs is not None:
        title_parts.append(f"UPDRS={updrs}")
    if med is not None:
        title_parts.append(f"Med={med}")
    title = "  |  ".join(title_parts)

    safe_name = f"{dataset_name}_{subject_key}_{trial_key}".replace("/", "_")
    output_path = OUTPUT_DIR / f"{safe_name}.mp4"

    create_animation(joints, fps, title, output_path, max_frames=max_frames)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="CARE-PD Gait Animation Generator")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g. BMCLab, 3DGait)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject key (e.g. SUB01, P001)")
    parser.add_argument("--trial", type=str, default=None,
                        help="Trial key (optional, uses first trial if not given)")
    parser.add_argument("--all", action="store_true",
                        help="Generate one animation per dataset")
    parser.add_argument("--max-frames", type=int, default=600,
                        help="Max frames per animation (default: 600)")
    args = parser.parse_args()

    dataset_files = {
        "3DGait": "3DGait.pkl", "BMCLab": "BMCLab.pkl", "DNE": "DNE.pkl",
        "E-LC": "E-LC.pkl", "KUL-DT-T": "KUL-DT-T.pkl", "PD-GaM": "PD-GaM.pkl",
        "T-LTC": "T-LTC.pkl", "T-SDU-PD": "T-SDU-PD.pkl", "T-SDU": "T-SDU.pkl",
    }

    if args.all:
        # Generate one sample animation per dataset
        for ds_name, ds_file in sorted(dataset_files.items()):
            path = DATA_DIR / ds_file
            if not path.exists():
                continue
            print(f"\n[{ds_name}]")
            with open(path, "rb") as f:
                data = pickle.load(f)
            subj = list(data.keys())[0]
            trial = list(data[subj].keys())[0]
            process_trial(ds_name, data, subj, trial, args.max_frames)
    else:
        # Single dataset mode
        ds_name = args.dataset or "BMCLab"
        ds_file = dataset_files.get(ds_name)
        if not ds_file:
            print(f"Unknown dataset: {ds_name}")
            print(f"Available: {', '.join(sorted(dataset_files.keys()))}")
            return

        path = DATA_DIR / ds_file
        print(f"Loading {ds_name}...")
        with open(path, "rb") as f:
            data = pickle.load(f)

        subj = args.subject or list(data.keys())[0]
        if subj not in data:
            print(f"Subject '{subj}' not found. Available: {list(data.keys())}")
            return

        trial_key = args.trial or list(data[subj].keys())[0]
        if trial_key not in data[subj]:
            print(f"Trial '{trial_key}' not found. Available: {list(data[subj].keys())}")
            return

        process_trial(ds_name, data, subj, trial_key, args.max_frames)

    print(f"\nAll animations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
