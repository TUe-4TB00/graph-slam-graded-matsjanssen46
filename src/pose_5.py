import copy
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))


def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))

    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )

    return graph, initial_estimate


def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))

    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )

    return graph


def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result


def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_score = float("inf")
    best_total_marginals = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            graph_trial = copy.deepcopy(graph)
            estimate_trial = copy.deepcopy(initial_estimate)

            graph_trial, estimate_trial = add_pose(graph_trial, estimate_trial, pose_5)

            result = optimize(graph_trial, estimate_trial)

            graph_trial = add_landmark_measurement(graph_trial, result, pose_5, landmark)

            result = optimize(graph_trial, estimate_trial)

            marginals = gtsam.Marginals(graph_trial, result)

            chosen_score = marginals.marginalCovariance(L(landmark)).sum()

            total_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if chosen_score < best_score:
                best_score = chosen_score
                best_pose = pose_name
                best_landmark = landmark
                best_total_marginals = total_marginals

    return best_pose, best_landmark, best_total_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_error = float("inf")

    true_poses = {
        1: gtsam.Pose2(0.0, 0.0, 0.0),
        2: gtsam.Pose2(2.0, 0.0, 0.0),
        3: gtsam.Pose2(4.0, 0.0, 0.0),
    }

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            graph_trial = copy.deepcopy(graph)
            estimate_trial = copy.deepcopy(initial_estimate)

            graph_trial, estimate_trial = add_pose(graph_trial, estimate_trial, pose_5)

            result = optimize(graph_trial, estimate_trial)

            graph_trial = add_landmark_measurement(graph_trial, result, pose_5, landmark)

            result = optimize(graph_trial, estimate_trial)

            list_of_errors = []

            for i in [1, 2, 3]:
                estimated_pose = result.atPose2(X(i))
                true_pose = true_poses[i]

                dx = estimated_pose.x() - true_pose.x()
                dy = estimated_pose.y() - true_pose.y()
                dtheta = estimated_pose.theta() - true_pose.theta()

                error = np.sqrt(dx**2 + dy**2 + dtheta**2)
                list_of_errors.append(error)

            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_error:
                best_error = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_error