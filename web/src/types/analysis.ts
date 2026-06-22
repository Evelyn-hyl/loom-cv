export interface PerspectiveAnalysis {
    vanishingPoints : VanishingPoint[];
}

export interface VanishingPoint {
    x : number;
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