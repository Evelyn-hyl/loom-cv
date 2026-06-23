export interface PerspectiveAnalysis {
    vanishingPoints : VanishingPoint[];
}

export interface VanishingPoint {
    x: number;
    y: number;
    inliers: Segment[];
}

export interface Segment {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    length: number;
}

export interface PoseEstimate {
    nose: Keypoint;
    leftEye: Keypoint;
    rightEye: Keypoint;
    leftEar: Keypoint;
    rightEar: Keypoint;
    leftShoulder: Keypoint;
    rightShoulder: Keypoint;
    leftElbow: Keypoint;
    rightElbow: Keypoint;
    leftWrist: Keypoint;
    rightWrist: Keypoint;
    leftHip: Keypoint;
    rightHip: Keypoint;
    leftKnee: Keypoint;
    rightKnee: Keypoint;
    leftAnkle: Keypoint;
    rightAnkle: Keypoint;
}

export interface Keypoint {
    x: number;
    y: number;
    confidence: number;
}