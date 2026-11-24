/**
 * *********************************************************
 *
 * @file: mppi_controller.cpp
 * @brief: Contains the Model Predictive Path Integral (MPPI) local controller
 * @version: 1.0
 *
 * ********************************************************
 */
#include <pluginlib/class_list_macros.h>

#include "common/util/log.h"
#include "controller/mppi_controller.h"

PLUGINLIB_EXPORT_CLASS(rmp::controller::MPPIController,
                       nav_core::BaseLocalPlanner)

namespace rmp::controller {
MPPIController::MPPIController()
  : initialized_(false), goal_reached_(false), tf_(nullptr) {
}

MPPIController::MPPIController(std::string name, tf2_ros::Buffer* tf,
                               costmap_2d::Costmap2DROS* costmap_ros)
  : MPPIController() {
  initialize(name, tf, costmap_ros);
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
  int implementation_stage = 1;
  private_nh.param("implementation_stage", implementation_stage, 1);

  tf_ = tf;
  costmap_ros_ = costmap_ros;
  goal_reached_ = false;
  global_plan_.clear();
  initialized_ = true;

  R_INFO << "MPPI Controller initialized in stage " << implementation_stage
         << " (safe-stop mode).";
}

bool MPPIController::setPlan(
    const std::vector<geometry_msgs::PoseStamped>& orig_global_plan) {
  if (!initialized_) {
    R_ERROR << "MPPI Controller has not been initialized; cannot accept a plan.";
    return false;
  }

  if (orig_global_plan.empty()) {
    global_plan_.clear();
    goal_reached_ = false;
    R_WARN << "MPPI Controller rejected an empty global plan.";
    return false;
  }

  global_plan_ = orig_global_plan;
  goal_reached_ = false;

  R_INFO << "MPPI Controller received a new global plan with "
         << global_plan_.size() << " poses in frame '"
         << global_plan_.front().header.frame_id << "'.";
  return true;
}

bool MPPIController::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
  // Always initialize every Twist component to a safe stop before validation.
  cmd_vel = geometry_msgs::Twist();

  if (!initialized_) {
    ROS_ERROR_THROTTLE(1.0,
                       "MPPI Controller has not been initialized; commanding zero velocity.");
    return false;
  }

  if (global_plan_.empty()) {
    ROS_WARN_THROTTLE(1.0,
                      "MPPI Controller has no global plan; commanding zero velocity.");
    return false;
  }

  ROS_INFO_THROTTLE(5.0,
                    "MPPI Controller stage 1 is active; commanding zero velocity.");
  return true;
}

bool MPPIController::isGoalReached() {
  if (!initialized_) {
    ROS_ERROR_THROTTLE(1.0, "MPPI Controller has not been initialized.");
    return false;
  }

  return goal_reached_;
}
}  // namespace rmp::controller
