import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_landmark_measurement(graph, initial_estimate, result):
    pose4 = result.atPose2(X(4))
    landmark2 = result.atPoint2(L(2))

    bearing = pose4.bearing(landmark2)
    distance = pose4.range(landmark2)

    graph.add(
        gtsam.BearingRangeFactor2D(
            X(4),
            L(2),
            bearing,
            distance,
            MEASUREMENT_NOISE
        )
    )

    return graph