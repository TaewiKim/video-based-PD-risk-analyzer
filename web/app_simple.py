"""
Simple Gait Analysis Web Server (Mock Mode)
============================================
For testing UI without MediaPipe dependency.
"""

import os
import json
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import random

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
DATA_FOLDER = BASE_DIR / 'data'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, DATA_FOLDER]:
    folder.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_mock_analysis():
    """Generate mock analysis results matching frontend expected structure."""
    n_segments = random.randint(2, 4)
    segments = []
    analysis_results = []
    current_time = random.uniform(1, 3)

    for i in range(n_segments):
        duration = round(random.uniform(4, 8), 1)
        end_time = round(current_time + duration, 1)
        speed = round(random.uniform(0.9, 1.3), 3)
        risk = random.choice(['Low', 'Low', 'Moderate'])

        segments.append({
            'segment_id': i + 1,
            'start_time': round(current_time, 1),
            'end_time': end_time,
            'duration': duration,
            'direction': random.choice(['away', 'toward']),
            'meets_min_duration': duration >= 3.0
        })

        analysis_results.append({
            'segment_id': i + 1,
            'start_time': round(current_time, 1),
            'end_time': end_time,
            'duration': duration,
            'walking_speed': speed,
            'stride_length': round(random.uniform(0.9, 1.3), 3),
            'cadence': round(random.uniform(95, 115), 1),
            'step_width': round(random.uniform(0.08, 0.15), 3),
            'stride_time_cv': round(random.uniform(2, 8), 2),
            'arm_swing_asymmetry': round(random.uniform(0.05, 0.25), 3),
            'arm_swing_amplitude': round(random.uniform(15, 35), 1),
            'step_time_asymmetry': round(random.uniform(0.02, 0.1), 3),
            'pd_risk_score': round(random.uniform(0.1, 0.4), 3),
            'pd_risk_level': risk,
        })

        current_time = end_time + random.uniform(2, 5)

    def make_param_stats(mean, std_ratio=0.1):
        std = mean * std_ratio
        return {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'cv': round(std / mean * 100, 1) if mean > 0 else 0,
            'min': round(mean - std * 1.5, 4),
            'max': round(mean + std * 1.5, 4),
            'ci_95': [round(mean - std * 1.96, 4), round(mean + std * 1.96, 4)],
            'n': n_segments
        }

    total_walking_time = sum(s['duration'] for s in segments)

    return {
        'success': True,
        'mode': 'mock',
        'message': 'Mock analysis (MediaPipe not available)',
        'walking': {
            'detected': True,
            'n_segments': n_segments,
            'total_duration': round(total_walking_time, 1),
            'segments': segments
        },
        'analysis_results': analysis_results,
        'statistical_analysis': {
            'analysis_quality': random.choice(['Good', 'Excellent']),
            'n_segments_total': n_segments,
            'n_segments_analyzed': n_segments,
            'total_walking_time_sec': round(total_walking_time, 1),
            'reliability_score': round(random.uniform(0.7, 0.95), 2),
            'min_segment_duration_sec': 3.0,
            'parameters': {
                'walking_speed': make_param_stats(random.uniform(1.0, 1.2)),
                'stride_length': make_param_stats(random.uniform(1.0, 1.2)),
                'cadence': make_param_stats(random.uniform(100, 110)),
                'step_width': make_param_stats(random.uniform(0.1, 0.12)),
                'stride_time_cv': make_param_stats(random.uniform(3, 5)),
                'arm_swing_asymmetry': make_param_stats(random.uniform(0.1, 0.15)),
                'arm_swing_amplitude': make_param_stats(random.uniform(20, 30)),
                'step_time_asymmetry': make_param_stats(random.uniform(0.04, 0.08)),
                'pd_risk_score': make_param_stats(random.uniform(0.15, 0.25)),
            }
        },
        'clinical_interpretation': {
            'overall_assessment': 'Normal gait pattern',
            'risk_level': 'Low',
            'confidence': round(random.uniform(0.7, 0.9), 2)
        }
    }


def generate_mock_symptoms():
    """Generate mock PD symptom analysis matching frontend expected structure."""
    duration = round(random.uniform(10, 30), 1)
    walking_duration = round(duration * random.uniform(0.4, 0.7), 1)
    resting_duration = round(duration - walking_duration, 1)

    def make_metric_stat(mean_val, std_ratio=0.15):
        std = mean_val * std_ratio
        return {
            'mean': round(mean_val, 4),
            'std': round(std, 4),
            'min': round(mean_val - std * 1.5, 4),
            'max': round(mean_val + std * 1.5, 4),
            'ci_lower': round(mean_val - std * 1.96, 4),
            'ci_upper': round(mean_val + std * 1.96, 4),
            'n': random.randint(2, 5)
        }

    def make_tremor_symptom():
        severity = random.choice(['normal', 'mild', 'moderate'])
        n_segments = random.randint(2, 5)
        return {
            'summary': {
                'overall_severity': severity,
                'n_segments': n_segments,
                'metrics_stats': {
                    'dominant_frequency': make_metric_stat(random.uniform(4, 6)),
                    'tremor_amplitude': make_metric_stat(random.uniform(0.5, 2.0)),
                    'freeze_index': make_metric_stat(random.uniform(0.1, 0.5)),
                    'tremor_regularity': make_metric_stat(random.uniform(0.3, 0.8)),
                    'score': make_metric_stat(random.uniform(0, 2)),
                    'pd_power_ratio': make_metric_stat(random.uniform(0.1, 0.4)),
                }
            },
            'results': []
        }

    def make_bradykinesia_symptom():
        severity = random.choice(['normal', 'mild', 'moderate'])
        n_segments = random.randint(2, 5)
        return {
            'summary': {
                'overall_severity': severity,
                'n_segments': n_segments,
                'metrics_stats': {
                    'movement_score': make_metric_stat(random.uniform(0, 2)),
                    'avg_velocity': make_metric_stat(random.uniform(50, 150)),
                    'decrement_score': make_metric_stat(random.uniform(0, 1)),
                    'hesitation_ratio': make_metric_stat(random.uniform(0.05, 0.2)),
                    'blink_rate': make_metric_stat(random.uniform(10, 20)),
                    'facial_movement': make_metric_stat(random.uniform(0.3, 0.8)),
                    'velocity_cv': make_metric_stat(random.uniform(10, 30)),
                }
            },
            'results': []
        }

    def make_posture_symptom():
        severity = random.choice(['normal', 'mild', 'moderate'])
        n_segments = random.randint(2, 5)
        return {
            'summary': {
                'overall_severity': severity,
                'n_segments': n_segments,
                'metrics_stats': {
                    'updrs_posture_score': make_metric_stat(random.uniform(0, 2)),
                    'trunk_forward_angle': make_metric_stat(random.uniform(5, 20)),
                    'lateral_angle': make_metric_stat(random.uniform(2, 10)),
                    'head_drop_angle': make_metric_stat(random.uniform(5, 15)),
                    'sway_index': make_metric_stat(random.uniform(0.5, 2.0)),
                    'lateral_lean': make_metric_stat(random.uniform(5, 20)),
                }
            },
            'results': [
                {
                    'metrics': {
                        'has_pisa_syndrome': False,
                        'severity_reasons': []
                    }
                }
            ]
        }

    def make_fog_symptom():
        severity = random.choice(['normal', 'mild'])
        n_segments = random.randint(2, 5)
        n_transitions = n_segments * 2  # Each walking segment has initiation + termination
        return {
            'summary': {
                'overall_severity': severity,
                'n_segments': n_segments,
                'transition_count': n_transitions,
                'initiation_count': n_segments,
                'termination_count': n_segments,
                'walking_segments': n_segments,
                'metrics_stats': {
                    'freeze_index': make_metric_stat(random.uniform(0.1, 0.5)),
                    'stride_cv': make_metric_stat(random.uniform(5, 15)),
                    'cadence': make_metric_stat(random.uniform(90, 120)),
                    'step_asymmetry': make_metric_stat(random.uniform(0.05, 0.15)),
                    'freeze_ratio': make_metric_stat(random.uniform(0, 0.1)),
                    'has_festination': False,
                    'ankle_hip_ratio': make_metric_stat(random.uniform(0.8, 1.2)),
                    'trembling_ratio': make_metric_stat(random.uniform(0, 0.1)),
                }
            },
            'transitions_detected': n_transitions,
            'transitions': [
                {'type': 'initiation', 'transition_type': 'standing_to_walking', 'boundary_time': i * 5.0}
                for i in range(n_segments)
            ] + [
                {'type': 'termination', 'transition_type': 'walking_to_standing', 'boundary_time': i * 5.0 + 3.0}
                for i in range(n_segments)
            ],
            'results': [
                {
                    'metrics': {
                        'severity_reasons': []
                    }
                }
            ]
        }

    def make_gait_symptom():
        severity = random.choice(['normal', 'mild', 'moderate'])
        n_segments = random.randint(2, 5)
        return {
            'summary': {
                'overall_severity': severity,
                'n_segments': n_segments,
                'metrics_stats': {
                    'walking_speed': make_metric_stat(random.uniform(0.8, 1.2)),
                    'stride_length': make_metric_stat(random.uniform(0.9, 1.3)),
                    'cadence': make_metric_stat(random.uniform(90, 120)),
                    'stride_cv': make_metric_stat(random.uniform(0.03, 0.08)),
                    'step_width': make_metric_stat(random.uniform(0.08, 0.15)),
                    'arm_swing_asymmetry': make_metric_stat(random.uniform(0.05, 0.25)),
                    'arm_swing_amplitude': make_metric_stat(random.uniform(15, 35)),
                    'step_time_asymmetry': make_metric_stat(random.uniform(0.02, 0.1)),
                    'stride_time_cv': make_metric_stat(random.uniform(0.02, 0.08)),
                    'double_support_time': make_metric_stat(random.uniform(0.2, 0.4)),
                    'swing_time_asymmetry': make_metric_stat(random.uniform(0.02, 0.08)),
                    'pd_risk_score': make_metric_stat(random.uniform(0.1, 0.4)),
                }
            },
            'results': []
        }

    # Generate gait analysis results (comprehensive)
    n_segments = random.randint(2, 4)
    walking_speed = round(random.uniform(0.9, 1.25), 3)
    stride_length = round(random.uniform(1.0, 1.3), 3)
    cadence = round(random.uniform(95, 115), 1)
    step_count = int(cadence * walking_duration / 60)

    # Reference ranges from literature
    REFERENCE_RANGES = {
        'walking_speed': {
            'unit': 'm/s',
            'healthy_mean': 1.21, 'healthy_std': 0.19,
            'pd_mean': 0.86, 'pd_std': 0.30,
            'pd_threshold': 0.55, 'direction': 'lower',
            'normal_range': [1.0, 1.4], 'mild_range': [0.7, 1.0], 'severe_range': [0, 0.7],
            'source': 'Mirelman et al. 2019'
        },
        'stride_length': {
            'unit': 'm',
            'healthy_mean': 1.30, 'healthy_std': 0.15,
            'pd_mean': 0.95, 'pd_std': 0.25,
            'pd_threshold': 0.60, 'direction': 'lower',
            'normal_range': [1.1, 1.5], 'mild_range': [0.8, 1.1], 'severe_range': [0, 0.8],
            'source': 'Hausdorff et al. 2001'
        },
        'cadence': {
            'unit': 'steps/min',
            'healthy_mean': 110, 'healthy_std': 10,
            'pd_mean': 100, 'pd_std': 15,
            'pd_threshold': 90, 'direction': 'lower',
            'normal_range': [100, 120], 'mild_range': [80, 100], 'severe_range': [0, 80],
            'source': 'Winter 1991'
        },
        'stride_time_cv': {
            'unit': '%',
            'healthy_mean': 2.0, 'healthy_std': 0.5,
            'pd_mean': 4.5, 'pd_std': 2.0,
            'pd_threshold': 2.6, 'direction': 'higher',
            'normal_range': [0, 2.5], 'mild_range': [2.5, 5], 'severe_range': [5, 100],
            'source': 'Hausdorff 2005'
        },
        'arm_swing_asymmetry': {
            'unit': '%',
            'healthy_mean': 0.05, 'healthy_std': 0.03,
            'pd_mean': 0.20, 'pd_std': 0.10,
            'pd_threshold': 0.10, 'direction': 'higher',
            'normal_range': [0, 0.10], 'mild_range': [0.10, 0.25], 'severe_range': [0.25, 1],
            'source': 'Lewek et al. 2010'
        },
        'step_time_asymmetry': {
            'unit': '%',
            'healthy_mean': 0.03, 'healthy_std': 0.02,
            'pd_mean': 0.08, 'pd_std': 0.04,
            'pd_threshold': 0.05, 'direction': 'higher',
            'normal_range': [0, 0.05], 'mild_range': [0.05, 0.10], 'severe_range': [0.10, 1],
            'source': 'Yogev et al. 2007'
        },
        'step_width': {
            'unit': 'm',
            'healthy_mean': 0.10, 'healthy_std': 0.03,
            'pd_mean': 0.08, 'pd_std': 0.02,
            'pd_threshold': 0.05, 'direction': 'lower',
            'normal_range': [0.08, 0.15], 'mild_range': [0.05, 0.08], 'severe_range': [0, 0.05],
            'source': 'Hollman et al. 2011'
        },
        'double_support_time': {
            'unit': '%',
            'healthy_mean': 0.20, 'healthy_std': 0.05,
            'pd_mean': 0.30, 'pd_std': 0.08,
            'pd_threshold': 0.25, 'direction': 'higher',
            'normal_range': [0.15, 0.25], 'mild_range': [0.25, 0.35], 'severe_range': [0.35, 1],
            'source': 'Morris et al. 1998'
        },
    }

    # Symptom-specific reference ranges
    SYMPTOM_REFERENCES = {
        'tremor': {
            'dominant_frequency': {'unit': 'Hz', 'pd_range': [4, 6], 'normal': '<3 or >8', 'source': 'Jankovic 2008'},
            'tremor_amplitude': {'unit': 'mm', 'normal_max': 0.5, 'mild': 2.0, 'severe': 5.0, 'source': 'UPDRS'},
            'pd_power_ratio': {'unit': '%', 'normal_max': 0.2, 'pd_threshold': 0.4, 'source': 'Bhatia et al. 2018'},
        },
        'bradykinesia': {
            'blink_rate': {'unit': '/min', 'healthy_mean': 17, 'pd_mean': 12, 'pd_threshold': 15, 'source': 'Karson 1983'},
            'movement_score': {'unit': 'UPDRS', 'normal': 0, 'mild': 1, 'moderate': 2, 'severe': '3-4', 'source': 'MDS-UPDRS'},
            'velocity_reduction': {'unit': '%', 'normal_max': 10, 'pd_threshold': 30, 'source': 'Espay et al. 2009'},
        },
        'posture': {
            'trunk_forward_angle': {'unit': '°', 'normal_max': 10, 'camptocormia': 45, 'source': 'Doherty et al. 2011'},
            'lateral_angle': {'unit': '°', 'normal_max': 5, 'pisa_syndrome': 10, 'source': 'Tinazzi et al. 2015'},
            'head_drop_angle': {'unit': '°', 'normal_max': 15, 'anterocollis': 45, 'source': 'Kashihara et al. 2006'},
        },
        'fog': {
            'freeze_index': {'unit': '', 'normal_max': 1.0, 'fog_threshold': 2.0, 'source': 'Moore et al. 2008'},
            'stride_cv': {'unit': '%', 'normal_max': 3, 'fog_risk': 8, 'source': 'Plotnik et al. 2008'},
            'festination_power': {'unit': '%', 'normal_max': 0.1, 'festination': 0.3, 'source': 'Nieuwboer et al. 2001'},
        },
    }

    # Generate indicator assessments
    stride_cv_val = round(random.uniform(0.02, 0.05), 3)
    arm_asym_val = round(random.uniform(0.05, 0.15), 3)
    step_asym_val = round(random.uniform(0.02, 0.06), 3)

    def assess_indicator(value, ref):
        threshold = ref['pd_threshold']
        direction = ref['direction']
        if direction == 'higher':
            is_abnormal = value > threshold
            severity = min((value - threshold) / threshold, 1.0) if is_abnormal else 0
        else:
            is_abnormal = value < threshold
            severity = min((threshold - value) / threshold, 1.0) if is_abnormal else 0
        return {
            'value': value,
            'threshold': threshold,
            'direction': direction,
            'is_abnormal': is_abnormal,
            'severity': round(severity, 3),
            'status': 'abnormal' if is_abnormal else 'normal',
            'healthy_mean': ref['healthy_mean'],
            'pd_mean': ref['pd_mean'],
        }

    gait_analysis = {
        'success': True,
        'classification': random.choice(['Healthy', 'Mild PD-like', 'Normal']),
        'pd_risk_score': round(random.uniform(10, 35), 1),
        'confidence': round(random.uniform(0.75, 0.95), 2),
        'n_segments': n_segments,
        'total_walking_time': round(walking_duration, 1),
        'reference_ranges': REFERENCE_RANGES,
        'biomarkers': {
            'walking_speed': walking_speed,
            'stride_length': stride_length,
            'stride_cv': stride_cv_val,
            'step_width': round(random.uniform(0.08, 0.14), 3),
            'arm_swing_asymmetry': arm_asym_val,
            'arm_swing_amplitude': round(random.uniform(18, 32), 1),
            'step_time_asymmetry': step_asym_val,
            'stride_time_cv': stride_cv_val,
            'double_support_time': round(random.uniform(0.20, 0.28), 3),
            'swing_time_asymmetry': round(random.uniform(0.02, 0.05), 3),
            'walk_ratio': round(stride_length / cadence * 100, 4),
            'pd_risk_score': round(random.uniform(0.1, 0.35), 3),
            'stability_score': round(random.uniform(0.7, 0.95), 2),
        },
        'gait_metrics': {
            'walking_speed': walking_speed,
            'stride_length': stride_length,
            'cadence': cadence,
            'step_count': step_count,
            'step_width': round(random.uniform(0.08, 0.14), 3),
            'step_length_left': round(stride_length / 2 * random.uniform(0.95, 1.05), 3),
            'step_length_right': round(stride_length / 2 * random.uniform(0.95, 1.05), 3),
            'stride_time': round(60 / cadence * 2, 3),
            'step_time_left': round(60 / cadence * random.uniform(0.95, 1.05), 3),
            'step_time_right': round(60 / cadence * random.uniform(0.95, 1.05), 3),
            'swing_time': round(0.4 * 60 / cadence, 3),
            'stance_time': round(0.6 * 60 / cadence, 3),
            'double_support_time': round(0.1 * 60 / cadence, 3),
        },
        'variability': {
            'stride_time_cv': round(random.uniform(2, 6), 2),
            'step_time_cv': round(random.uniform(2, 5), 2),
            'stride_length_cv': round(random.uniform(3, 7), 2),
            'step_width_cv': round(random.uniform(15, 30), 2),
        },
        'asymmetry': {
            'step_time_asymmetry': step_asym_val,
            'step_length_asymmetry': round(random.uniform(0.02, 0.06), 3),
            'swing_time_asymmetry': round(random.uniform(0.02, 0.06), 3),
            'arm_swing_asymmetry': arm_asym_val,
        },
        'arm_swing': {
            'amplitude_left': round(random.uniform(20, 35), 1),
            'amplitude_right': round(random.uniform(20, 35), 1),
            'asymmetry': arm_asym_val,
            'coordination': round(random.uniform(0.7, 0.95), 2),
        },
        'pd_assessment': {
            'overall': {
                'risk_level': 'Low',
                'risk_score': round(random.uniform(0.1, 0.3), 2),
                'abnormal_count': random.randint(0, 2),
                'primary_abnormal': random.randint(0, 1),
            },
            'indicators': {
                'stride_variability': {
                    'name': 'Stride Time CV',
                    'mean': stride_cv_val,
                    'std': round(stride_cv_val * 0.2, 4),
                    'threshold': 0.026,
                    'abnormal_pct': round(random.uniform(0, 30), 1),
                    'p_value': round(random.uniform(0.1, 0.9), 3),
                    'statistically_abnormal': False,
                },
                'arm_swing': {
                    'name': 'Arm Swing Asymmetry',
                    'mean': arm_asym_val,
                    'std': round(arm_asym_val * 0.2, 4),
                    'threshold': 0.10,
                    'abnormal_pct': round(random.uniform(0, 40), 1),
                    'p_value': round(random.uniform(0.05, 0.8), 3),
                    'statistically_abnormal': arm_asym_val > 0.10,
                },
                'step_timing': {
                    'name': 'Step Time Asymmetry',
                    'mean': step_asym_val,
                    'std': round(step_asym_val * 0.2, 4),
                    'threshold': 0.05,
                    'abnormal_pct': round(random.uniform(0, 25), 1),
                    'p_value': round(random.uniform(0.1, 0.9), 3),
                    'statistically_abnormal': step_asym_val > 0.05,
                },
            },
            'speed_assessment': assess_indicator(walking_speed, REFERENCE_RANGES['walking_speed']),
            'stride_assessment': assess_indicator(stride_length, REFERENCE_RANGES['stride_length']),
        },
        'segments': [
            {
                'segment_id': i + 1,
                'start_time': round(i * 5 + random.uniform(0, 2), 1),
                'end_time': round(i * 5 + random.uniform(4, 6), 1),
                'walking_speed': round(walking_speed * random.uniform(0.9, 1.1), 3),
                'stride_length': round(stride_length * random.uniform(0.9, 1.1), 3),
                'cadence': round(cadence * random.uniform(0.95, 1.05), 1),
                'pd_risk_level': random.choice(['Low', 'Low', 'Moderate']),
            }
            for i in range(n_segments)
        ],
        'fog_transitions': [
            {'type': 'initiation', 'transition_type': 'standing_to_walking', 'boundary_time': i * 5.0 + 0.5}
            for i in range(n_segments)
        ] + [
            {'type': 'termination', 'transition_type': 'walking_to_standing', 'boundary_time': i * 5.0 + 4.5}
            for i in range(n_segments)
        ],
        'fog_transition_count': n_segments * 2,
        'clinical_interpretation': {
            'overall_assessment': 'Normal gait pattern with minor variability',
            'risk_level': 'Low',
            'key_findings': [
                f'Walking speed {walking_speed:.2f} m/s (healthy: >1.0 m/s, PD risk: <0.55 m/s)',
                f'Stride length {stride_length:.2f} m (healthy: >1.1 m, PD risk: <0.60 m)',
                f'Arm swing asymmetry {arm_asym_val*100:.1f}% (healthy: <10%, PD risk: >10%)',
                f'Stride time CV {stride_cv_val*100:.1f}% (healthy: <2.6%, PD risk: >2.6%)',
            ],
            'recommendations': [
                'Continue regular physical activity',
                'Monitor for changes over time',
            ],
            'comparison_notes': [
                f'Speed is {((walking_speed - 0.86) / 0.86 * 100):.0f}% above PD mean (0.86 m/s)',
                f'Stride variability within healthy range (<2.6%)',
            ]
        }
    }

    # Add symptom reference ranges to the response
    symptom_references = SYMPTOM_REFERENCES

    gait_analysis['symptom_references'] = symptom_references

    return {
        'success': True,
        'mode': 'mock',
        'n_persons': 1,
        'video_info': {
            'fps': 30,
            'total_frames': int(duration * 30),
            'duration': duration,
            'resolution': [1280, 720]
        },
        'gait_analysis': gait_analysis,
        'persons': [
            {
                'person_id': 'Person_1',
                'duration': duration,
                'start_frame': 0,
                'end_frame': int(duration * 30),
                'activity_breakdown': {
                    'walking': walking_duration,
                    'resting': resting_duration,
                    'task': 0,
                    'standing': 0
                },
                'activity_segments': [
                    {
                        'activity_type': 'walking',
                        'start_time': 0,
                        'end_time': walking_duration,
                        'confidence': 0.85
                    },
                    {
                        'activity_type': 'resting',
                        'start_time': walking_duration,
                        'end_time': duration,
                        'confidence': 0.90
                    }
                ],
                'symptoms': {
                    'tremor': make_tremor_symptom(),
                    'bradykinesia': make_bradykinesia_symptom(),
                    'posture': make_posture_symptom(),
                    'gait': make_gait_symptom(),
                    'fog': make_fog_symptom(),
                }
            }
        ]
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        return jsonify({
            'success': True,
            'filename': filename,
            'video_url': url_for('serve_video', filename=filename)
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run mock gait analysis."""
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    # Return mock results
    results = generate_mock_analysis()
    results['filename'] = filename

    # Save results
    result_file = RESULTS_FOLDER / f"{filename.rsplit('.', 1)[0]}_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return jsonify(results)


@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    """Run mock PD symptom analysis."""
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    # Return mock results
    results = generate_mock_symptoms()
    results['filename'] = filename

    # Save results
    result_file = RESULTS_FOLDER / f"{filename.rsplit('.', 1)[0]}_symptoms.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return jsonify(results)


@app.route('/register-user', methods=['POST'])
def register_user():
    """Mock user registration."""
    data = request.json
    name = data.get('name')

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    return jsonify({
        'success': True,
        'user': {
            'id': str(uuid.uuid4())[:8],
            'name': name,
            'registered_at': '2024-01-01T00:00:00'
        }
    })


@app.route('/users', methods=['GET'])
def list_users():
    """Return mock user list."""
    return jsonify([
        {'id': 'user001', 'name': 'Test User 1'},
        {'id': 'user002', 'name': 'Test User 2'},
    ])


@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route('/reference-data')
def get_reference_data():
    return jsonify({
        'healthy_young': {'mean': 1.24, 'std': 0.18, 'n': 24},
        'healthy_older': {'mean': 1.21, 'std': 0.19, 'n': 18},
        'pd_off': {'mean': 0.86, 'std': 0.30, 'n': 23},
        'pd_on': {'mean': 1.02, 'std': 0.28, 'n': 25},
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Gait Analysis Web Server (MOCK MODE)")
    print("=" * 60)
    print("\n⚠️  Running in mock mode - MediaPipe not available")
    print("    Analysis results are simulated for UI testing\n")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
