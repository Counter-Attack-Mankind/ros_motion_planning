/**
 * *********************************************************
 *
 * @file: mppi_controller.cpp
 * @brief: Contains the Model Predictive Path Integral (MPPI) local controller
 * @version: 2.0
 *
 * ********************************************************
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

#include <base_local_planner/costmap_model.h>
#include <costmap_2d/cost_values.h>
#include <costmap_2d/layered_costmap.h>
#include <nav_msgs/Odometry.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "common/geometry/angles.h"
#include "common/util/log.h"
#include "controller/mppi_controller.h"

PLUGINLIB_EXPORT_CLASS(rmp::controller::MPPIController,
                       nav_core::BaseLocalPlanner)

namespace rmp::controller {
namespace {
constexpr double kNumericalEpsilon = 1e-12;
constexpr double kMaximumExponent = 700.0;
constexpr int kMaximumFallbackCandidates = 5;
}  // namespace

MPPIController::MPPIController()
  : initialized_(false)
  , goal_reached_(false)
  , has_goal_(false)
  , footprint_logged_(false)
  , visualization_enabled_(true)
  , visualization_active_(false)
  , tf_(nullptr)
  , sample_count_(200)
  , prediction_steps_(60)
  , optimization_iterations_(1)
  , random_seed_(42)
  , temporal_noise_enabled_(true)
  , temporal_noise_rho_(0.9)
  , temporal_noise_v_enabled_(true)
  , temporal_noise_rho_v_(0.8)
  , safety_fallback_enabled_(true)
  , safety_fallback_max_candidates_(5)
  , noise_std_v_(0.2)
  , noise_std_omega_(0.5)
  , temperature_(1.0)
  , path_distance_weight_(8.0)
  , path_heading_weight_(0.5)
  , terminal_progress_weight_(6.0)
  , control_v_weight_(0.2)
  , control_omega_weight_(0.1)
  , control_delta_v_weight_(1.0)
  , control_delta_omega_weight_(0.2)
  , obstacle_cost_weight_(10.0)
  , max_control_delta_v_(0.05)
  , max_control_delta_omega_(0.3)
  , collision_check_resolution_(0.025)
  , treat_unknown_as_collision_(true)
  , collision_safety_horizon_(1.0)
  , visualization_frequency_(5.0)
  , visualization_sample_count_(100)
  , visualization_rejected_count_(50)
  , goal_reset_distance_(0.5)
  , goal_reset_angle_(0.5)
  , random_generator_(random_seed_) {
}

MPPIController::MPPIController(std::string name, tf2_ros::Buffer* tf,
                               costmap_2d::Costmap2DROS* costmap_ros)
  : MPPIController() {
  initialize(std::move(name), tf, costmap_ros);
}

void MPPIController::initialize(std::string name, tf2_ros::Buffer* tf,
                                costmap_2d::Costmap2DROS* costmap_ros) {
  if (initialized_) {
    R_WARN << "MPPI Controller has already been initialized.";
    return;
  }

  if (tf == nullptr || costmap_ros == nullptr) {
    R_ERROR << "MPPI Controller initialization failed: TF buffer or costmap is null.";
    return;
  }

  ros::NodeHandle private_nh("~/" + name);
  if (!_loadParameters(private_nh)) {
    R_ERROR << "MPPI Controller initialization failed: invalid parameters.";
    return;
  }

  tf_ = tf;
  costmap_ros_ = costmap_ros;
  goal_reached_ = false;
  has_goal_ = false;
  global_plan_.clear();
  random_generator_.seed(static_cast<std::mt19937::result_type>(random_seed_));
  _resetControlSequence();
  if (visualization_enabled_) {
    sampled_trajectories_publisher_ = private_nh.advertise<
        visualization_msgs::Marker>("sampled_trajectories", 1);
    rejected_trajectories_publisher_ = private_nh.advertise<
        visualization_msgs::Marker>("rejected_trajectories", 1);
    optimized_trajectory_publisher_ =
        private_nh.advertise<nav_msgs::Path>("optimized_trajectory", 1);
  }
  initialized_ = true;

  R_INFO << "MPPI Controller initialized: samples=" << sample_count_
         << ", prediction steps=" << prediction_steps_
         << ", iterations=" << optimization_iterations_
         << ", dt=" << control_dt_
         << " s, collision interpolation=" << collision_check_resolution_
         << " m, unknown space="
         << (treat_unknown_as_collision_ ? "blocked" : "allowed")
         << ", verified safety horizon=" << collision_safety_horizon_ << " s"
         << ", temporal noise="
         << (temporal_noise_enabled_ ? "enabled" : "disabled")
         << ", temporal linear noise="
         << (temporal_noise_v_enabled_ ? "enabled" : "disabled")
         << ", safety fallback="
         << (safety_fallback_enabled_ ? "enabled" : "disabled")
         << ", visualization="
         << (visualization_enabled_ ? "enabled" : "disabled") << ".";
}

bool MPPIController::_loadParameters(ros::NodeHandle& private_nh) {
  private_nh.param("sample_count", sample_count_, sample_count_);
  private_nh.param("prediction_steps", prediction_steps_, prediction_steps_);
  private_nh.param("optimization_iterations", optimization_iterations_,
                   optimization_iterations_);
  private_nh.param("random_seed", random_seed_, random_seed_);
  private_nh.param("temporal_noise_enabled", temporal_noise_enabled_,
                   temporal_noise_enabled_);
  private_nh.param("temporal_noise_rho", temporal_noise_rho_,
                   temporal_noise_rho_);
  private_nh.param("temporal_noise_v_enabled", temporal_noise_v_enabled_,
                   temporal_noise_v_enabled_);
  private_nh.param("temporal_noise_rho_v", temporal_noise_rho_v_,
                   temporal_noise_rho_v_);
  private_nh.param("safety_fallback_enabled", safety_fallback_enabled_,
                   safety_fallback_enabled_);
  private_nh.param("safety_fallback_max_candidates",
                   safety_fallback_max_candidates_,
                   safety_fallback_max_candidates_);

  private_nh.param("noise_std_v", noise_std_v_, noise_std_v_);
  private_nh.param("noise_std_omega", noise_std_omega_, noise_std_omega_);
  private_nh.param("temperature", temperature_, temperature_);

  private_nh.param("path_distance_weight", path_distance_weight_,
                   path_distance_weight_);
  private_nh.param("path_heading_weight", path_heading_weight_,
                   path_heading_weight_);
  private_nh.param("terminal_progress_weight", terminal_progress_weight_,
                   terminal_progress_weight_);
  private_nh.param("control_v_weight", control_v_weight_, control_v_weight_);
  private_nh.param("control_omega_weight", control_omega_weight_,
                   control_omega_weight_);
  private_nh.param("control_delta_v_weight", control_delta_v_weight_,
                   control_delta_v_weight_);
  private_nh.param("control_delta_omega_weight", control_delta_omega_weight_,
                   control_delta_omega_weight_);
  private_nh.param("obstacle_cost_weight", obstacle_cost_weight_,
                   obstacle_cost_weight_);

  private_nh.param("max_control_delta_v", max_control_delta_v_,
                   max_control_delta_v_);
  private_nh.param("max_control_delta_omega", max_control_delta_omega_,
                   max_control_delta_omega_);
  private_nh.param("collision_check_resolution", collision_check_resolution_,
                   collision_check_resolution_);
  private_nh.param("treat_unknown_as_collision", treat_unknown_as_collision_,
                   treat_unknown_as_collision_);
  private_nh.param("collision_safety_horizon", collision_safety_horizon_,
                   collision_safety_horizon_);
  private_nh.param("visualization_enabled", visualization_enabled_,
                   visualization_enabled_);
  private_nh.param("visualization_frequency", visualization_frequency_,
                   visualization_frequency_);
  private_nh.param("visualization_sample_count", visualization_sample_count_,
                   visualization_sample_count_);
  private_nh.param("visualization_rejected_count",
                   visualization_rejected_count_,
                   visualization_rejected_count_);
  private_nh.param("goal_reset_distance", goal_reset_distance_,
                   goal_reset_distance_);
  private_nh.param("goal_reset_angle", goal_reset_angle_, goal_reset_angle_);

  const bool dimensions_valid = sample_count_ > 1 && prediction_steps_ > 0 &&
                                optimization_iterations_ > 0 &&
                                safety_fallback_max_candidates_ >= 0 &&
                                safety_fallback_max_candidates_ <=
                                    kMaximumFallbackCandidates &&
                                (!safety_fallback_enabled_ ||
                                 safety_fallback_max_candidates_ > 0);
  const bool sampling_valid = std::isfinite(noise_std_v_) &&
                              std::isfinite(noise_std_omega_) &&
                              noise_std_v_ > 0.0 && noise_std_omega_ > 0.0 &&
                              std::isfinite(temperature_) &&
                              temperature_ > kNumericalEpsilon &&
                              std::isfinite(temporal_noise_rho_) &&
                              temporal_noise_rho_ >= 0.0 &&
                              temporal_noise_rho_ < 1.0 &&
                              std::isfinite(temporal_noise_rho_v_) &&
                              temporal_noise_rho_v_ >= 0.0 &&
                              temporal_noise_rho_v_ < 1.0;
  const bool limits_valid = std::isfinite(max_control_delta_v_) &&
                            std::isfinite(max_control_delta_omega_) &&
                            std::isfinite(control_dt_) && control_dt_ > 0.0 &&
                            max_control_delta_v_ > 0.0 &&
                            max_control_delta_omega_ > 0.0 &&
                            std::isfinite(collision_check_resolution_) &&
                            collision_check_resolution_ > 0.0 &&
                            std::isfinite(collision_safety_horizon_) &&
                            collision_safety_horizon_ > 0.0 &&
                            std::isfinite(visualization_frequency_) &&
                            visualization_frequency_ > 0.0 &&
                            visualization_sample_count_ >= 0 &&
                            visualization_rejected_count_ >= 0 &&
                            std::isfinite(goal_reset_distance_) &&
                            std::isfinite(goal_reset_angle_) &&
                            goal_reset_distance_ >= 0.0 &&
                            goal_reset_angle_ >= 0.0;
  const bool weights_valid =
      std::isfinite(path_distance_weight_) && path_distance_weight_ >= 0.0 &&
      std::isfinite(path_heading_weight_) && path_heading_weight_ >= 0.0 &&
      std::isfinite(terminal_progress_weight_) &&
      terminal_progress_weight_ >= 0.0 && std::isfinite(control_v_weight_) &&
      control_v_weight_ >= 0.0 && std::isfinite(control_omega_weight_) &&
      control_omega_weight_ >= 0.0 &&
      std::isfinite(control_delta_v_weight_) &&
      control_delta_v_weight_ >= 0.0 &&
      std::isfinite(control_delta_omega_weight_) &&
      control_delta_omega_weight_ >= 0.0 &&
      std::isfinite(obstacle_cost_weight_) && obstacle_cost_weight_ >= 0.0;

  return dimensions_valid && sampling_valid && limits_valid && weights_valid;
}

bool MPPIController::setPlan(
    const std::vector<geometry_msgs::PoseStamped>& orig_global_plan) {
  if (!initialized_) {
    R_ERROR << "MPPI Controller has not been initialized; cannot accept a plan.";
    return false;
  }

  if (orig_global_plan.empty()) {
    global_plan_.clear();
    has_goal_ = false;
    goal_reached_ = false;
    _resetControlSequence();
    _clearVisualization();
    R_WARN << "MPPI Controller rejected an empty global plan.";
    return false;
  }

  if (orig_global_plan.front().header.frame_id.empty() ||
      orig_global_plan.back().header.frame_id.empty()) {
    global_plan_.clear();
    has_goal_ = false;
    goal_reached_ = false;
    _clearVisualization();
    R_WARN << "MPPI Controller rejected a global plan with an empty frame ID.";
    return false;
  }

  const geometry_msgs::PoseStamped& new_goal = orig_global_plan.back();
  const bool goal_changed =
      !has_goal_ || _isGoalSignificantlyDifferent(new_goal);
  const bool reset_controls = goal_reached_ || goal_changed;

  global_plan_ = orig_global_plan;
  goal_pose_ = new_goal;
  has_goal_ = true;
  if (goal_changed) {
    goal_reached_ = false;
  }

  if (reset_controls) {
    _resetControlSequence();
  }
  if (goal_changed) {
    _clearVisualization();
  }

  ROS_INFO_THROTTLE(
      2.0,
      "MPPI Controller received a global plan with %zu poses in frame '%s'; "
      "control warm start %s.",
      global_plan_.size(), global_plan_.front().header.frame_id.c_str(),
      reset_controls ? "reset" : "retained");
  return true;
}

bool MPPIController::_isGoalSignificantlyDifferent(
    const geometry_msgs::PoseStamped& new_goal) const {
  if (!has_goal_ || new_goal.header.frame_id != goal_pose_.header.frame_id) {
    return true;
  }

  const double position_difference =
      std::hypot(new_goal.pose.position.x - goal_pose_.pose.position.x,
                 new_goal.pose.position.y - goal_pose_.pose.position.y);
  const double angle_difference = std::fabs(common::geometry::normalizeAngle(
      tf2::getYaw(new_goal.pose.orientation) -
      tf2::getYaw(goal_pose_.pose.orientation)));

  return position_difference > goal_reset_distance_ ||
         angle_difference > goal_reset_angle_;
}

bool MPPIController::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
  cmd_vel = geometry_msgs::Twist();
  const auto cycle_start = std::chrono::steady_clock::now();

  if (!initialized_) {
    ROS_ERROR_THROTTLE(
        1.0,
        "MPPI Controller has not been initialized; commanding zero velocity.");
    return false;
  }

  if (global_plan_.empty() || !has_goal_) {
    _clearVisualization();
    ROS_WARN_THROTTLE(
        1.0, "MPPI Controller has no global plan; commanding zero velocity.");
    return false;
  }

  nav_msgs::Odometry base_odom;
  odom_helper_->getOdom(base_odom);
  const Control current_control{ base_odom.twist.twist.linear.x,
                                 base_odom.twist.twist.angular.z };
  if (!std::isfinite(current_control.v) ||
      !std::isfinite(current_control.omega)) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(1.0,
                       "MPPI Controller received non-finite odometry velocity.");
    return false;
  }

  geometry_msgs::PoseStamped robot_pose_local;
  if (!costmap_ros_->getRobotPose(robot_pose_local)) {
    _clearVisualization();
    ROS_WARN_THROTTLE(1.0, "MPPI Controller could not obtain the robot pose.");
    return false;
  }
  if (robot_pose_local.header.frame_id.empty()) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(
        1.0, "MPPI Controller received a robot pose with no frame ID.");
    return false;
  }

  const State initial_state{ robot_pose_local.pose.position.x,
                             robot_pose_local.pose.position.y,
                             tf2::getYaw(robot_pose_local.pose.orientation) };
  if (!std::isfinite(initial_state.x) || !std::isfinite(initial_state.y) ||
      !std::isfinite(initial_state.yaw)) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(1.0,
                       "MPPI Controller received a non-finite robot pose.");
    return false;
  }

  costmap_2d::Costmap2D* local_costmap = costmap_ros_->getCostmap();
  costmap_2d::LayeredCostmap* layered_costmap =
      costmap_ros_->getLayeredCostmap();
  const std::vector<geometry_msgs::Point> footprint =
      costmap_ros_->getRobotFootprint();
  if (local_costmap == nullptr || layered_costmap == nullptr ||
      footprint.size() < 3) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(
        1.0,
        "MPPI Controller has no valid runtime local-costmap footprint.");
    return false;
  }

  const double inscribed_radius = layered_costmap->getInscribedRadius();
  const double circumscribed_radius = layered_costmap->getCircumscribedRadius();
  if (!std::isfinite(inscribed_radius) || inscribed_radius < 0.0 ||
      !std::isfinite(circumscribed_radius) || circumscribed_radius < 0.0) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(
        1.0, "MPPI Controller received invalid local-costmap footprint radii.");
    return false;
  }

  if (!footprint_logged_) {
    R_INFO << "MPPI collision checking uses " << footprint.size()
           << " runtime footprint points in frame '"
           << robot_pose_local.header.frame_id << "', inscribed radius="
           << inscribed_radius << " m, circumscribed radius="
           << circumscribed_radius << " m, costmap resolution="
           << local_costmap->getResolution() << " m.";
    footprint_logged_ = true;
  }

  // robot_pose_local and every predicted state remain in the local costmap's
  // global frame; CostmapModel rotates/translates this robot-frame footprint.
  base_local_planner::CostmapModel collision_model(*local_costmap);
  if (_checkStateCollision(initial_state, collision_model, footprint,
                           inscribed_radius, circumscribed_radius) !=
      CollisionCheckResult::kFree) {
    ROS_WARN_THROTTLE(
        1.0,
        "MPPI Controller robot footprint is in collision, unknown space, or "
        "outside the local costmap; commanding zero velocity.");
    _resetControlSequence();
    _clearVisualization();
    return false;
  }

  const std::string plan_frame = global_plan_.front().header.frame_id;
  if (plan_frame.empty()) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(1.0, "MPPI Controller received a plan with no frame ID.");
    return false;
  }

  geometry_msgs::PoseStamped robot_pose_plan;
  geometry_msgs::PoseStamped goal_pose_local;
  try {
    if (robot_pose_local.header.frame_id == plan_frame) {
      robot_pose_plan = robot_pose_local;
    } else {
      transformPose(tf_, plan_frame, robot_pose_local, robot_pose_plan);
    }

    if (goal_pose_.header.frame_id == robot_pose_local.header.frame_id) {
      goal_pose_local = goal_pose_;
    } else {
      geometry_msgs::PoseStamped latest_goal = goal_pose_;
      latest_goal.header.stamp = ros::Time(0);
      transformPose(tf_, robot_pose_local.header.frame_id, latest_goal,
                    goal_pose_local);
    }
  } catch (const tf2::TransformException& exception) {
    _clearVisualization();
    ROS_WARN_THROTTLE(1.0, "MPPI Controller TF failure: %s", exception.what());
    return false;
  }

  if (shouldRotateToGoal(robot_pose_local, goal_pose_local)) {
    _resetControlSequence();
    _clearVisualization();
    const double yaw_error = common::geometry::normalizeAngle(
        tf2::getYaw(goal_pose_local.pose.orientation) -
        tf2::getYaw(robot_pose_local.pose.orientation));

    if (!shouldRotateToPath(std::fabs(yaw_error))) {
      goal_reached_ = true;
      R_INFO << "MPPI Controller reached the goal.";
      return true;
    }

    cmd_vel.angular.z =
        angularRegularization(current_control.omega, yaw_error / control_dt_);
    if (!std::isfinite(cmd_vel.angular.z)) {
      cmd_vel = geometry_msgs::Twist();
      return false;
    }

    State rotated_state = initial_state;
    _predictState(&rotated_state, Control{ 0.0, cmd_vel.angular.z });
    if (_checkMotionCollision(initial_state, rotated_state, collision_model,
                              footprint, inscribed_radius,
                              circumscribed_radius) !=
        CollisionCheckResult::kFree) {
      cmd_vel = geometry_msgs::Twist();
      ROS_WARN_THROTTLE(
          1.0,
          "MPPI Controller rejected an unsafe final-goal rotation command.");
      return false;
    }
    return true;
  }
  goal_reached_ = false;

  std::vector<geometry_msgs::PoseStamped> pruned_plan = prune(robot_pose_plan);
  if (pruned_plan.empty()) {
    _clearVisualization();
    ROS_WARN_THROTTLE(1.0,
                      "MPPI Controller produced an empty pruned reference path.");
    return false;
  }

  std::vector<geometry_msgs::PoseStamped> reference_plan;
  if (!_transformPlan(pruned_plan, robot_pose_local.header.frame_id,
                      &reference_plan)) {
    _clearVisualization();
    return false;
  }

  ros::Time visualization_stamp;
  bool collect_visualization = false;
  if (visualization_enabled_) {
    visualization_stamp = ros::Time::now();
    collect_visualization = _visualizationDue(visualization_stamp);
  }
  std::vector<std::vector<State>> sampled_trajectories;
  std::vector<std::vector<State>> rejected_trajectories;
  auto* sampled_trajectories_output =
      collect_visualization ? &sampled_trajectories : nullptr;
  auto* rejected_trajectories_output =
      collect_visualization ? &rejected_trajectories : nullptr;
  std::size_t valid_sample_count = 0;
  std::size_t collision_rejection_count = 0;
  std::size_t safety_rejection_count = 0;
  std::size_t unverified_sample_count = 0;
  std::vector<FallbackCandidate> fallback_candidates;
  auto* fallback_candidates_output =
      safety_fallback_enabled_ ? &fallback_candidates : nullptr;
  if (!_optimizeControlSequence(initial_state, current_control, reference_plan,
                                collision_model, footprint, inscribed_radius,
                                circumscribed_radius, &valid_sample_count,
                                &collision_rejection_count,
                                &safety_rejection_count,
                                &unverified_sample_count,
                                fallback_candidates_output,
                                sampled_trajectories_output,
                                rejected_trajectories_output) ||
      nominal_controls_.empty()) {
    _resetControlSequence();
    _clearVisualization();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cycle_start)
            .count();
    ROS_WARN_THROTTLE(
        1.0,
        "MPPI Controller has no feasible weighted control sequence "
        "(valid=%zu/%d, collision rejected=%zu, unverified-prefix "
        "rejected=%zu, unverified-tail=%zu, cycle=%.3f ms); commanding zero "
        "velocity and returning false to move_base for replanning/recovery.",
        valid_sample_count, sample_count_ * optimization_iterations_,
        collision_rejection_count, safety_rejection_count,
        unverified_sample_count, elapsed_ms);
    return false;
  }

  std::vector<Control> execution_controls;
  if (!_prepareExecutionControls(nominal_controls_, current_control, &cmd_vel,
                                 &execution_controls)) {
    _clearVisualization();
    ROS_ERROR_THROTTLE(1.0,
                       "MPPI Controller generated a non-finite velocity command.");
    return false;
  }

  bool fallback_used = false;
  std::size_t fallback_attempts = 0;
  std::size_t fallback_rank = 0;
  double fallback_cost = 0.0;
  const bool weighted_result_safe = _isControlSequenceCollisionFree(
      initial_state, execution_controls, collision_model, footprint,
      inscribed_radius, circumscribed_radius);
  if (!weighted_result_safe && safety_fallback_enabled_) {
    for (std::size_t index = 0; index < fallback_candidates.size(); ++index) {
      ++fallback_attempts;
      geometry_msgs::Twist fallback_command;
      std::vector<Control> fallback_execution_controls;
      if (!_prepareExecutionControls(fallback_candidates[index].controls,
                                     current_control, &fallback_command,
                                     &fallback_execution_controls)) {
        continue;
      }
      if (!_isControlSequenceCollisionFree(
              initial_state, fallback_execution_controls, collision_model,
              footprint, inscribed_radius, circumscribed_radius)) {
        continue;
      }

      cmd_vel = fallback_command;
      execution_controls = std::move(fallback_execution_controls);
      nominal_controls_ = execution_controls;
      fallback_used = true;
      fallback_rank = index + 1;
      fallback_cost = fallback_candidates[index].cost;
      ROS_WARN_THROTTLE(
          1.0,
          "MPPI weighted result was unsafe; safety fallback selected cost "
          "rank %zu/%zu (cost=%.3f, attempts=%zu).",
          fallback_rank, fallback_candidates.size(), fallback_cost,
          fallback_attempts);
      break;
    }
  }

  if (!weighted_result_safe && !fallback_used) {
    cmd_vel = geometry_msgs::Twist();
    _resetControlSequence();
    _clearVisualization();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cycle_start)
            .count();
    ROS_WARN_THROTTLE(
        1.0,
        "MPPI weighted result is unsafe; safety fallback %s "
        "(attempted=%zu/%zu, valid=%zu/%d, collision rejected=%zu, "
        "unverified-prefix rejected=%zu, unverified-tail=%zu, cycle=%.3f ms); "
        "commanding zero velocity.",
        safety_fallback_enabled_ ? "failed" : "disabled",
        fallback_attempts, fallback_candidates.size(),
        valid_sample_count, sample_count_ * optimization_iterations_,
        collision_rejection_count, safety_rejection_count,
        unverified_sample_count, elapsed_ms);
    return false;
  }

  if (collect_visualization) {
    _publishVisualization(initial_state, execution_controls,
                          sampled_trajectories, rejected_trajectories,
                          robot_pose_local.header.frame_id,
                          visualization_stamp);
  }
  _shiftControlSequence();

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cycle_start)
          .count();
  ROS_INFO_THROTTLE(
      2.0,
      "MPPI cycle %.3f ms, valid samples %zu/%d, collision rejected %zu, "
      "unverified-prefix rejected %zu, unverified-tail %zu, command v=%.3f "
      "m/s, omega=%.3f rad/s, solution=%s.",
      elapsed_ms, valid_sample_count,
      sample_count_ * optimization_iterations_,
      collision_rejection_count, safety_rejection_count,
      unverified_sample_count, cmd_vel.linear.x, cmd_vel.angular.z,
      fallback_used ? "fallback" : "weighted");
  return true;
}

bool MPPIController::_transformPlan(
    const std::vector<geometry_msgs::PoseStamped>& plan,
    const std::string& target_frame,
    std::vector<geometry_msgs::PoseStamped>* transformed_plan) const {
  transformed_plan->clear();
  transformed_plan->reserve(plan.size());

  try {
    for (const auto& pose : plan) {
      if (pose.header.frame_id.empty()) {
        ROS_ERROR_THROTTLE(
            1.0, "MPPI Controller reference path contains an empty frame ID.");
        transformed_plan->clear();
        return false;
      }

      if (pose.header.frame_id == target_frame) {
        transformed_plan->push_back(pose);
      } else {
        geometry_msgs::PoseStamped latest_pose = pose;
        latest_pose.header.stamp = ros::Time(0);
        geometry_msgs::PoseStamped transformed_pose;
        transformPose(tf_, target_frame, latest_pose, transformed_pose);
        transformed_plan->push_back(std::move(transformed_pose));
      }
    }
  } catch (const tf2::TransformException& exception) {
    transformed_plan->clear();
    ROS_WARN_THROTTLE(1.0, "MPPI Controller path TF failure: %s",
                      exception.what());
    return false;
  }

  return !transformed_plan->empty();
}

bool MPPIController::_optimizeControlSequence(
    const State& initial_state, const Control& current_control,
    const std::vector<geometry_msgs::PoseStamped>& reference_plan,
    base_local_planner::CostmapModel& collision_model,
    const std::vector<geometry_msgs::Point>& footprint,
    double inscribed_radius, double circumscribed_radius,
    std::size_t* valid_sample_count,
    std::size_t* collision_rejection_count,
    std::size_t* safety_rejection_count,
    std::size_t* unverified_sample_count,
    std::vector<FallbackCandidate>* fallback_candidates,
    std::vector<std::vector<State>>* sampled_trajectories,
    std::vector<std::vector<State>>* rejected_trajectories) {
  *valid_sample_count = 0;
  *collision_rejection_count = 0;
  *safety_rejection_count = 0;
  *unverified_sample_count = 0;
  if (fallback_candidates != nullptr) {
    fallback_candidates->clear();
    fallback_candidates->reserve(
        static_cast<std::size_t>(safety_fallback_max_candidates_));
  }
  const std::size_t required_verified_steps = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(collision_safety_horizon_ / control_dt_)));
  const std::size_t available_sample_count =
      static_cast<std::size_t>(sample_count_) *
      static_cast<std::size_t>(optimization_iterations_);
  if (sampled_trajectories != nullptr) {
    sampled_trajectories->clear();
    sampled_trajectories->reserve(std::min(
        static_cast<std::size_t>(visualization_sample_count_),
        available_sample_count));
  }
  if (rejected_trajectories != nullptr) {
    rejected_trajectories->clear();
    rejected_trajectories->reserve(std::min(
        static_cast<std::size_t>(visualization_rejected_count_),
        available_sample_count));
  }
  if (reference_plan.empty() || nominal_controls_.size() !=
                                    static_cast<std::size_t>(prediction_steps_)) {
    return false;
  }

  std::vector<double> cumulative_distance(reference_plan.size(), 0.0);
  for (std::size_t index = 1; index < reference_plan.size(); ++index) {
    cumulative_distance[index] =
        cumulative_distance[index - 1] +
        std::hypot(reference_plan[index].pose.position.x -
                       reference_plan[index - 1].pose.position.x,
                   reference_plan[index].pose.position.y -
                       reference_plan[index - 1].pose.position.y);
  }

  std::normal_distribution<double> normal_distribution(0.0, 1.0);
  const double temporal_noise_scale = std::sqrt(std::max(
      0.0, 1.0 - temporal_noise_rho_ * temporal_noise_rho_));
  const double temporal_noise_scale_v = std::sqrt(std::max(
      0.0, 1.0 - temporal_noise_rho_v_ * temporal_noise_rho_v_));
  std::size_t final_valid_sample_count = 0;
  std::size_t total_valid_sample_count = 0;
  std::size_t total_collision_rejection_count = 0;
  std::size_t total_safety_rejection_count = 0;
  std::size_t total_unverified_sample_count = 0;

  for (int iteration = 0; iteration < optimization_iterations_; ++iteration) {
    std::vector<std::vector<Control>> sampled_controls(
        sample_count_, std::vector<Control>(prediction_steps_));
    std::vector<double> sample_costs(
        sample_count_, std::numeric_limits<double>::infinity());
    std::size_t iteration_valid_sample_count = 0;

    for (int sample = 0; sample < sample_count_; ++sample) {
      State state = initial_state;
      Control previous_control = current_control;
      double previous_noise_v = 0.0;
      double previous_noise_omega = 0.0;
      const bool record_sample =
          (sampled_trajectories != nullptr &&
           sampled_trajectories->size() <
               static_cast<std::size_t>(visualization_sample_count_)) ||
          (rejected_trajectories != nullptr &&
           rejected_trajectories->size() <
               static_cast<std::size_t>(visualization_rejected_count_));
      std::vector<State> sampled_states;
      if (record_sample) {
        sampled_states.reserve(static_cast<std::size_t>(prediction_steps_ + 1));
        sampled_states.push_back(initial_state);
      }
      std::size_t nearest_index =
          _nearestReferenceIndex(state, reference_plan, 0);
      double total_cost = 0.0;
      double obstacle_cost_sum = 0.0;
      bool sample_valid = true;
      bool sample_has_unverified_tail = false;

      for (int step = 0; step < prediction_steps_; ++step) {
        const double independent_noise_v =
            sample == 0 ? 0.0 :
                          noise_std_v_ * normal_distribution(random_generator_);
        double noise_v = independent_noise_v;
        if (sample != 0 && temporal_noise_v_enabled_ && step > 0) {
          noise_v = temporal_noise_rho_v_ * previous_noise_v +
                    temporal_noise_scale_v * independent_noise_v;
        }
        previous_noise_v = noise_v;
        const double independent_noise_omega =
            sample == 0 ?
                0.0 :
                noise_std_omega_ * normal_distribution(random_generator_);
        double noise_omega = independent_noise_omega;
        if (sample != 0 && temporal_noise_enabled_ && step > 0) {
          noise_omega = temporal_noise_rho_ * previous_noise_omega +
                        temporal_noise_scale * independent_noise_omega;
        }
        previous_noise_omega = noise_omega;

        Control sampled_control{
          nominal_controls_[step].v + noise_v,
          nominal_controls_[step].omega + noise_omega
        };
        sampled_control = _projectControl(sampled_control, previous_control);
        sampled_controls[sample][step] = sampled_control;

        const State previous_state = state;
        _predictState(&state, sampled_control);
        if (!std::isfinite(state.x) || !std::isfinite(state.y) ||
            !std::isfinite(state.yaw)) {
          sample_valid = false;
          break;
        }
        if (record_sample) {
          sampled_states.push_back(state);
        }
        const CollisionCheckResult collision_result = _checkMotionCollision(
            previous_state, state, collision_model, footprint,
            inscribed_radius, circumscribed_radius);
        if (collision_result == CollisionCheckResult::kCollision) {
          sample_valid = false;
          ++total_collision_rejection_count;
          if (rejected_trajectories != nullptr &&
              rejected_trajectories->size() < static_cast<std::size_t>(
                  visualization_rejected_count_)) {
            rejected_trajectories->push_back(std::move(sampled_states));
          }
          break;
        }
        if (collision_result == CollisionCheckResult::kOutsideMap) {
          // A map-external state is not declared collision-free. The verified
          // prefix is mandatory; an unverified tail keeps only the existing
          // path/control costs and is reconsidered as the rolling map moves.
          if (static_cast<std::size_t>(step) < required_verified_steps) {
            sample_valid = false;
            ++total_safety_rejection_count;
            break;
          }
          sample_has_unverified_tail = true;
        }

        const double normalized_obstacle_cost =
            collision_result == CollisionCheckResult::kOutsideMap ?
                1.0 :
                _normalizedObstacleCost(state, footprint);
        obstacle_cost_sum +=
            normalized_obstacle_cost * normalized_obstacle_cost;

        nearest_index =
            _nearestReferenceIndex(state, reference_plan, nearest_index);
        const double path_dx =
            state.x - reference_plan[nearest_index].pose.position.x;
        const double path_dy =
            state.y - reference_plan[nearest_index].pose.position.y;
        const double heading_error = common::geometry::normalizeAngle(
            state.yaw - _referenceHeading(reference_plan, nearest_index));
        const double delta_v = sampled_control.v - previous_control.v;
        const double delta_omega =
            sampled_control.omega - previous_control.omega;

        total_cost += path_distance_weight_ *
                          (path_dx * path_dx + path_dy * path_dy) +
                      path_heading_weight_ * (1.0 - std::cos(heading_error)) +
                      control_v_weight_ * sampled_control.v * sampled_control.v +
                      control_omega_weight_ * sampled_control.omega *
                          sampled_control.omega +
                      control_delta_v_weight_ * delta_v * delta_v +
                      control_delta_omega_weight_ * delta_omega * delta_omega;
        previous_control = sampled_control;
      }

      if (!sample_valid) {
        continue;
      }

      const double remaining_path_distance =
          cumulative_distance.back() - cumulative_distance[nearest_index];
      total_cost += terminal_progress_weight_ * remaining_path_distance;
      total_cost += obstacle_cost_weight_ * obstacle_cost_sum /
                    static_cast<double>(prediction_steps_);

      if (std::isfinite(total_cost)) {
        sample_costs[sample] = total_cost;
        ++iteration_valid_sample_count;
        if (fallback_candidates != nullptr) {
          const auto position = std::lower_bound(
              fallback_candidates->begin(), fallback_candidates->end(),
              total_cost,
              [](const FallbackCandidate& candidate, double cost) {
                return candidate.cost < cost;
              });
          if (fallback_candidates->size() < static_cast<std::size_t>(
                  safety_fallback_max_candidates_) ||
              position != fallback_candidates->end()) {
            fallback_candidates->insert(
                position,
                FallbackCandidate{ total_cost, sampled_controls[sample] });
            if (fallback_candidates->size() > static_cast<std::size_t>(
                    safety_fallback_max_candidates_)) {
              fallback_candidates->pop_back();
            }
          }
        }
        if (sample_has_unverified_tail) {
          ++total_unverified_sample_count;
        }
        if (sampled_trajectories != nullptr &&
            sampled_trajectories->size() < static_cast<std::size_t>(
                visualization_sample_count_)) {
          sampled_trajectories->push_back(std::move(sampled_states));
        }
      }
    }

    total_valid_sample_count += iteration_valid_sample_count;
    if (iteration_valid_sample_count == 0) {
      *valid_sample_count = total_valid_sample_count;
      *collision_rejection_count = total_collision_rejection_count;
      *safety_rejection_count = total_safety_rejection_count;
      *unverified_sample_count = total_unverified_sample_count;
      return false;
    }

    double minimum_cost = std::numeric_limits<double>::infinity();
    for (const double sample_cost : sample_costs) {
      if (std::isfinite(sample_cost)) {
        minimum_cost = std::min(minimum_cost, sample_cost);
      }
    }
    if (!std::isfinite(minimum_cost)) {
      *valid_sample_count = total_valid_sample_count;
      *collision_rejection_count = total_collision_rejection_count;
      *safety_rejection_count = total_safety_rejection_count;
      *unverified_sample_count = total_unverified_sample_count;
      return false;
    }

    std::vector<double> weights(sample_count_, 0.0);
    double weight_sum = 0.0;
    for (int sample = 0; sample < sample_count_; ++sample) {
      if (!std::isfinite(sample_costs[sample])) {
        continue;
      }

      const double exponent = std::clamp(
          -(sample_costs[sample] - minimum_cost) / temperature_,
          -kMaximumExponent, 0.0);
      const double weight = std::exp(exponent);
      if (std::isfinite(weight)) {
        weights[sample] = weight;
        weight_sum += weight;
      }
    }

    if (!std::isfinite(weight_sum) || weight_sum <= kNumericalEpsilon) {
      *valid_sample_count = total_valid_sample_count;
      *collision_rejection_count = total_collision_rejection_count;
      *safety_rejection_count = total_safety_rejection_count;
      *unverified_sample_count = total_unverified_sample_count;
      return false;
    }

    for (int step = 0; step < prediction_steps_; ++step) {
      double weighted_delta_v = 0.0;
      double weighted_delta_omega = 0.0;
      for (int sample = 0; sample < sample_count_; ++sample) {
        if (weights[sample] <= 0.0) {
          continue;
        }
        weighted_delta_v +=
            weights[sample] *
            (sampled_controls[sample][step].v - nominal_controls_[step].v);
        weighted_delta_omega +=
            weights[sample] *
            (sampled_controls[sample][step].omega -
             nominal_controls_[step].omega);
      }
      nominal_controls_[step].v += weighted_delta_v / weight_sum;
      nominal_controls_[step].omega += weighted_delta_omega / weight_sum;
    }

    _projectControlSequence(current_control);
    final_valid_sample_count = iteration_valid_sample_count;
  }

  *valid_sample_count = total_valid_sample_count;
  *collision_rejection_count = total_collision_rejection_count;
  *safety_rejection_count = total_safety_rejection_count;
  *unverified_sample_count = total_unverified_sample_count;
  return final_valid_sample_count > 0;
}

bool MPPIController::_visualizationDue(const ros::Time& stamp) const {
  if (!visualization_enabled_) {
    return false;
  }
  if (last_visualization_time_.isZero() ||
      stamp < last_visualization_time_) {
    return true;
  }
  return (stamp - last_visualization_time_).toSec() >=
         1.0 / visualization_frequency_;
}

void MPPIController::_publishVisualization(
    const State& initial_state,
    const std::vector<Control>& execution_controls,
    const std::vector<std::vector<State>>& sampled_trajectories,
    const std::vector<std::vector<State>>& rejected_trajectories,
    const std::string& frame_id, const ros::Time& stamp) {
  const nav_msgs::Path optimized_path =
      _buildOptimizedPath(initial_state, execution_controls, frame_id, stamp);
  if (optimized_path.poses.empty()) {
    _clearVisualization();
    ROS_WARN_THROTTLE(
        1.0, "MPPI Controller could not build a finite visualization path.");
    return;
  }

  _publishTrajectoryMarker(sampled_trajectories, "mppi_sampled_trajectories",
                           frame_id, stamp, 0.0F, 0.35F, 1.0F, 0.2F,
                           &sampled_trajectories_publisher_);
  _publishTrajectoryMarker(rejected_trajectories,
                           "mppi_rejected_trajectories", frame_id, stamp,
                           1.0F, 0.0F, 0.0F, 0.35F,
                           &rejected_trajectories_publisher_);
  optimized_trajectory_publisher_.publish(optimized_path);

  last_visualization_time_ = stamp;
  visualization_frame_ = frame_id;
  visualization_active_ = true;
}

void MPPIController::_publishTrajectoryMarker(
    const std::vector<std::vector<State>>& trajectories,
    const std::string& marker_namespace, const std::string& frame_id,
    const ros::Time& stamp, float red, float green, float blue, float alpha,
    ros::Publisher* publisher) const {
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = marker_namespace;
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = trajectories.empty() ? visualization_msgs::Marker::DELETE :
                                         visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.01;
  marker.color.r = red;
  marker.color.g = green;
  marker.color.b = blue;
  marker.color.a = alpha;

  if (!trajectories.empty()) {
    std::size_t segment_count = 0;
    for (const auto& trajectory : trajectories) {
      if (trajectory.size() > 1) {
        segment_count += trajectory.size() - 1;
      }
    }
    marker.points.reserve(segment_count * 2);
    for (const auto& trajectory : trajectories) {
      for (std::size_t index = 1; index < trajectory.size(); ++index) {
        geometry_msgs::Point from;
        from.x = trajectory[index - 1].x;
        from.y = trajectory[index - 1].y;
        geometry_msgs::Point to;
        to.x = trajectory[index].x;
        to.y = trajectory[index].y;
        marker.points.push_back(from);
        marker.points.push_back(to);
      }
    }
  }
  publisher->publish(marker);
}

nav_msgs::Path MPPIController::_buildOptimizedPath(
    const State& initial_state,
    const std::vector<Control>& execution_controls,
    const std::string& frame_id, const ros::Time& stamp) const {
  nav_msgs::Path path;
  path.header.frame_id = frame_id;
  path.header.stamp = stamp;
  path.poses.reserve(execution_controls.size() + 1);

  State state = initial_state;
  for (std::size_t index = 0; index <= execution_controls.size(); ++index) {
    if (!std::isfinite(state.x) || !std::isfinite(state.y) ||
        !std::isfinite(state.yaw)) {
      path.poses.clear();
      return path;
    }

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = state.x;
    pose.pose.position.y = state.y;
    tf2::Quaternion orientation;
    orientation.setRPY(0.0, 0.0, state.yaw);
    pose.pose.orientation = tf2::toMsg(orientation);
    path.poses.push_back(std::move(pose));

    if (index < execution_controls.size()) {
      _predictState(&state, execution_controls[index]);
    }
  }
  return path;
}

void MPPIController::_clearVisualization() {
  if (!visualization_enabled_ || !visualization_active_) {
    return;
  }

  const ros::Time stamp = ros::Time::now();
  visualization_msgs::Marker marker;
  marker.header.frame_id = visualization_frame_;
  marker.header.stamp = stamp;
  marker.id = 0;
  marker.action = visualization_msgs::Marker::DELETE;

  marker.ns = "mppi_sampled_trajectories";
  sampled_trajectories_publisher_.publish(marker);
  marker.ns = "mppi_rejected_trajectories";
  rejected_trajectories_publisher_.publish(marker);

  nav_msgs::Path empty_path;
  empty_path.header.frame_id = visualization_frame_;
  empty_path.header.stamp = stamp;
  optimized_trajectory_publisher_.publish(empty_path);
  visualization_active_ = false;
  last_visualization_time_ = ros::Time(0);
}

MPPIController::CollisionCheckResult MPPIController::_checkStateCollision(
    const State& state,
    base_local_planner::CostmapModel& collision_model,
    const std::vector<geometry_msgs::Point>& footprint,
    double inscribed_radius, double circumscribed_radius) const {
  if (!std::isfinite(state.x) || !std::isfinite(state.y) ||
      !std::isfinite(state.yaw)) {
    return CollisionCheckResult::kCollision;
  }

  const costmap_2d::Costmap2D* local_costmap = costmap_ros_->getCostmap();
  unsigned int map_x = 0;
  unsigned int map_y = 0;
  if (local_costmap == nullptr ||
      !local_costmap->worldToMap(state.x, state.y, map_x, map_y)) {
    return local_costmap == nullptr ? CollisionCheckResult::kCollision :
                                      CollisionCheckResult::kOutsideMap;
  }
  const unsigned char center_cost = local_costmap->getCost(map_x, map_y);
  if (center_cost == costmap_2d::NO_INFORMATION &&
      treat_unknown_as_collision_) {
    return CollisionCheckResult::kCollision;
  }
  if (center_cost != costmap_2d::NO_INFORMATION &&
      center_cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
    return CollisionCheckResult::kCollision;
  }

  const double footprint_cost = collision_model.footprintCost(
      state.x, state.y, state.yaw, footprint, inscribed_radius,
      circumscribed_radius);
  if (!std::isfinite(footprint_cost)) {
    return CollisionCheckResult::kCollision;
  }
  if (footprint_cost >= 0.0) {
    return CollisionCheckResult::kFree;
  }

  // CostmapModel reports -3 when any footprint point is outside the map. This
  // is distinct from a collision so a long-horizon tail can remain explicitly
  // unverified while the near-term safety horizon remains mandatory.
  if (footprint_cost == -3.0) {
    return CollisionCheckResult::kOutsideMap;
  }
  if (footprint_cost == -2.0 && !treat_unknown_as_collision_) {
    return CollisionCheckResult::kFree;
  }
  return CollisionCheckResult::kCollision;
}

double MPPIController::_normalizedObstacleCost(
    const State& state,
    const std::vector<geometry_msgs::Point>& footprint) const {
  const costmap_2d::Costmap2D* local_costmap = costmap_ros_->getCostmap();
  if (local_costmap == nullptr || !std::isfinite(state.x) ||
      !std::isfinite(state.y) || !std::isfinite(state.yaw)) {
    return 1.0;
  }

  unsigned char maximum_cost = costmap_2d::FREE_SPACE;
  const auto include_cost = [&](double world_x, double world_y) {
    unsigned int map_x = 0;
    unsigned int map_y = 0;
    if (!local_costmap->worldToMap(world_x, world_y, map_x, map_y)) {
      return false;
    }
    maximum_cost =
        std::max(maximum_cost, local_costmap->getCost(map_x, map_y));
    return true;
  };

  if (!include_cost(state.x, state.y)) {
    return 1.0;
  }
  const double cosine = std::cos(state.yaw);
  const double sine = std::sin(state.yaw);
  for (const geometry_msgs::Point& footprint_point : footprint) {
    const double world_x = state.x + footprint_point.x * cosine -
                           footprint_point.y * sine;
    const double world_y = state.y + footprint_point.x * sine +
                           footprint_point.y * cosine;
    if (!include_cost(world_x, world_y)) {
      return 1.0;
    }
  }

  const double bounded_cost = std::min<double>(
      maximum_cost, costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
  return bounded_cost /
         static_cast<double>(costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
}

MPPIController::CollisionCheckResult MPPIController::_checkMotionCollision(
    const State& from, const State& to,
    base_local_planner::CostmapModel& collision_model,
    const std::vector<geometry_msgs::Point>& footprint,
    double inscribed_radius, double circumscribed_radius) const {
  const double translation = std::hypot(to.x - from.x, to.y - from.y);
  const double yaw_delta = common::geometry::normalizeAngle(to.yaw - from.yaw);
  const double rotation_sweep =
      std::fabs(yaw_delta) * circumscribed_radius;
  const double maximum_sweep = std::max(translation, rotation_sweep);
  const std::size_t interpolation_steps = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(maximum_sweep / collision_check_resolution_)));

  CollisionCheckResult motion_result = CollisionCheckResult::kFree;
  for (std::size_t index = 1; index <= interpolation_steps; ++index) {
    const double ratio = static_cast<double>(index) /
                         static_cast<double>(interpolation_steps);
    const State interpolated{
      from.x + ratio * (to.x - from.x),
      from.y + ratio * (to.y - from.y),
      common::geometry::normalizeAngle(from.yaw + ratio * yaw_delta)
    };
    const CollisionCheckResult state_result = _checkStateCollision(
        interpolated, collision_model, footprint, inscribed_radius,
        circumscribed_radius);
    if (state_result == CollisionCheckResult::kCollision) {
      return CollisionCheckResult::kCollision;
    }
    if (state_result == CollisionCheckResult::kOutsideMap) {
      motion_result = CollisionCheckResult::kOutsideMap;
    }
  }
  return motion_result;
}

bool MPPIController::_isControlSequenceCollisionFree(
    const State& initial_state, const std::vector<Control>& controls,
    base_local_planner::CostmapModel& collision_model,
    const std::vector<geometry_msgs::Point>& footprint,
    double inscribed_radius, double circumscribed_radius) const {
  State state = initial_state;
  const std::size_t required_verified_steps = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(collision_safety_horizon_ / control_dt_)));
  for (std::size_t index = 0; index < controls.size(); ++index) {
    const State previous_state = state;
    _predictState(&state, controls[index]);
    const CollisionCheckResult collision_result = _checkMotionCollision(
        previous_state, state, collision_model, footprint, inscribed_radius,
        circumscribed_radius);
    if (collision_result == CollisionCheckResult::kCollision ||
        (collision_result == CollisionCheckResult::kOutsideMap &&
         index < required_verified_steps)) {
      return false;
    }
  }
  return true;
}

void MPPIController::_predictState(State* state, const Control& control) const {
  state->x += control.v * std::cos(state->yaw) * control_dt_;
  state->y += control.v * std::sin(state->yaw) * control_dt_;
  state->yaw = common::geometry::normalizeAngle(
      state->yaw + control.omega * control_dt_);
}

MPPIController::Control MPPIController::_projectControl(
    const Control& control, const Control& previous_control) const {
  Control projected = control;
  projected.v = std::clamp(
      projected.v, previous_control.v - max_control_delta_v_,
      previous_control.v + max_control_delta_v_);
  projected.omega = std::clamp(
      projected.omega, previous_control.omega - max_control_delta_omega_,
      previous_control.omega + max_control_delta_omega_);

  projected.v = std::clamp(projected.v, config_.min_linear_velocity(),
                           config_.max_linear_velocity());
  projected.omega = std::clamp(projected.omega,
                               -config_.max_angular_velocity(),
                               config_.max_angular_velocity());
  return projected;
}

void MPPIController::_projectControlSequence(
    const Control& current_control) {
  Control previous_control = current_control;
  for (auto& control : nominal_controls_) {
    control = _projectControl(control, previous_control);
    previous_control = control;
  }
}

bool MPPIController::_prepareExecutionControls(
    const std::vector<Control>& source_controls,
    const Control& current_control, geometry_msgs::Twist* command,
    std::vector<Control>* execution_controls) {
  *command = geometry_msgs::Twist();
  execution_controls->clear();
  if (source_controls.empty()) {
    return false;
  }

  const Control desired_control =
      _projectControl(source_controls.front(), current_control);
  command->linear.x =
      linearRegularization(current_control.v, desired_control.v);
  command->angular.z =
      angularRegularization(current_control.omega, desired_control.omega);
  if (!std::isfinite(command->linear.x) ||
      !std::isfinite(command->angular.z)) {
    *command = geometry_msgs::Twist();
    return false;
  }

  *execution_controls = source_controls;
  execution_controls->front() =
      Control{ command->linear.x, command->angular.z };
  Control previous_control = execution_controls->front();
  for (std::size_t index = 1; index < execution_controls->size(); ++index) {
    (*execution_controls)[index] =
        _projectControl((*execution_controls)[index], previous_control);
    if (!std::isfinite((*execution_controls)[index].v) ||
        !std::isfinite((*execution_controls)[index].omega)) {
      *command = geometry_msgs::Twist();
      execution_controls->clear();
      return false;
    }
    previous_control = (*execution_controls)[index];
  }
  return true;
}

std::size_t MPPIController::_nearestReferenceIndex(
    const State& state,
    const std::vector<geometry_msgs::PoseStamped>& reference_plan,
    std::size_t search_begin) const {
  search_begin = std::min(search_begin, reference_plan.size() - 1);
  std::size_t nearest_index = search_begin;
  double minimum_squared_distance = std::numeric_limits<double>::infinity();

  for (std::size_t index = search_begin; index < reference_plan.size(); ++index) {
    const double dx = state.x - reference_plan[index].pose.position.x;
    const double dy = state.y - reference_plan[index].pose.position.y;
    const double squared_distance = dx * dx + dy * dy;
    if (squared_distance < minimum_squared_distance) {
      minimum_squared_distance = squared_distance;
      nearest_index = index;
    }
  }
  return nearest_index;
}

double MPPIController::_referenceHeading(
    const std::vector<geometry_msgs::PoseStamped>& reference_plan,
    std::size_t index) const {
  if (reference_plan.size() == 1) {
    return tf2::getYaw(reference_plan.front().pose.orientation);
  }

  std::size_t first_index = index;
  std::size_t second_index = std::min(index + 1, reference_plan.size() - 1);
  if (first_index == second_index) {
    first_index = second_index - 1;
  }

  const double dx = reference_plan[second_index].pose.position.x -
                    reference_plan[first_index].pose.position.x;
  const double dy = reference_plan[second_index].pose.position.y -
                    reference_plan[first_index].pose.position.y;
  if (std::hypot(dx, dy) <= kNumericalEpsilon) {
    return tf2::getYaw(reference_plan[index].pose.orientation);
  }
  return std::atan2(dy, dx);
}

void MPPIController::_resetControlSequence() {
  nominal_controls_.assign(static_cast<std::size_t>(prediction_steps_), Control{});
}

void MPPIController::_shiftControlSequence() {
  if (nominal_controls_.empty()) {
    return;
  }
  for (std::size_t index = 0; index + 1 < nominal_controls_.size(); ++index) {
    nominal_controls_[index] = nominal_controls_[index + 1];
  }
  nominal_controls_.back() = Control{};
}

bool MPPIController::isGoalReached() {
  if (!initialized_) {
    ROS_ERROR_THROTTLE(1.0, "MPPI Controller has not been initialized.");
    return false;
  }

  return goal_reached_;
}
}  // namespace rmp::controller
