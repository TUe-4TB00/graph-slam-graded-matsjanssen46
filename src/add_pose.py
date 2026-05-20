import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate):
    odometry = (
        gtsam.Pose2(0.0, 0.0, math.radians(45))
        .compose(gtsam.Pose2(2.0, 0.0, 0.0))
        .compose(gtsam.Pose2(0.0, 0.0, math.radians(45)))
    )

    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    initial_estimate.insert(
        X(4),
        gtsam.Pose2(5.414, 1.414, math.radians(90))
    )

    return graph, initial_estimate